import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange

from local_attention.local_attention import LocalAttention

# helper function


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# def l2norm(t):
#    return F.normalize(t, dim = -1)

# multi-head attention


class LocalMHA(Module):
    def __init__(
        self,
        *,
        dim,
        window_size,
        dim_head=64,
        heads=8,
        dropout=0.0,
        prenorm=True,
        # qk_rmsnorm = False,
        # qk_scale = 8,
        use_xpos=True,
        xpos_scale_base=None,
        exact_windowsize=None,
        **kwargs,
    ):
        super().__init__()
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if prenorm else None

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 2, bias=False)

        # self.qk_rmsnorm = qk_rmsnorm

        # if qk_rmsnorm:
        #     self.q_scale = nn.Parameter(torch.ones(dim_head))
        #     self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.window_size = window_size
        self.exact_windowsize = default(exact_windowsize, True)

        self.attn_fn = LocalAttention(
            dim=dim_head,
            window_size=window_size,
            causal=False,
            autopad=True,
            scale=None,
            exact_windowsize=self.exact_windowsize,
            use_xpos=use_xpos,
            xpos_scale_base=xpos_scale_base,
            **kwargs,
        )

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None):
        if exists(self.norm):
            x = self.norm(x)

        qk, v = self.to_qkv(x).chunk(2, dim=-1)
        qk, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (qk, v)
        )

        # if self.qk_rmsnorm:
        #     q, k = map(l2norm, (q, k))
        #     q = q * self.q_scale
        #     k = k * self.k_scale

        out = self.attn_fn(qk, qk, v, mask=mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return out


# feedforward


class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


def FeedForward(dim, mult=4, dropout=0.0):
    inner_dim = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


# main transformer class


class LocalTransformerEncoder(Module):
    def __init__(
        self,
        dim,
        depth,
        local_attn_window_size=512,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        use_xpos=False,
        xpos_scale_base=None,
        **kwargs,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(
                ModuleList(
                    [
                        LocalMHA(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            window_size=local_attn_window_size,
                            use_xpos=use_xpos,
                            xpos_scale_base=xpos_scale_base,
                            use_rotary_pos_emb=True,
                            prenorm=True,
                            shared_qk=True,
                            look_backward=1,
                            look_forward=1,
                            **kwargs,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x

        return self.norm(x)
