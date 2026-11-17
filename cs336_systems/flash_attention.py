"""Tiled FlashAttention-2 reference implementation using only PyTorch."""

import torch
from torch.autograd.function import once_differentiable


TILE_SIZE = 64


def _scores(q, k, query_start, key_start, is_causal):
    scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if is_causal:
        rows = torch.arange(query_start, query_start + q.shape[-2], device=q.device)
        cols = torch.arange(key_start, key_start + k.shape[-2], device=k.device)
        scores = scores.masked_fill(rows[:, None] < cols[None, :], -torch.inf)
    return scores


@torch.compile
def _backward(q, k, v, o, grad_o, lse, is_causal):
    # Equations 13-19: reconstruct probabilities one tile at a time.
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    o, grad_o = o.to(dtype), grad_o.to(dtype)
    d = (o * grad_o).sum(dim=-1)
    grad_q, grad_k, grad_v = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    scale = q.shape[-1] ** -0.5
    for i in range(0, q.shape[-2], TILE_SIZE):
        qi = q[..., i:i + TILE_SIZE, :]
        doi = grad_o[..., i:i + TILE_SIZE, :]
        li = lse[..., i:i + TILE_SIZE, None]
        di = d[..., i:i + TILE_SIZE, None]
        for j in range(0, k.shape[-2], TILE_SIZE):
            kj = k[..., j:j + TILE_SIZE, :]
            vj = v[..., j:j + TILE_SIZE, :]
            p = torch.exp(_scores(qi, kj, i, j, is_causal) - li)
            dp = doi @ vj.transpose(-2, -1)
            ds = p * (dp - di)
            grad_q[..., i:i + TILE_SIZE, :] += (ds @ kj) * scale
            grad_k[..., j:j + TILE_SIZE, :] += (ds.transpose(-2, -1) @ qi) * scale
            grad_v[..., j:j + TILE_SIZE, :] += p.transpose(-2, -1) @ doi
    return grad_q, grad_k, grad_v


class FlashAttention2(torch.autograd.Function):
    """Attention on (..., sequence, head_dim), with optional top-left causal mask.

    Only Q, K, V, O and row-wise logsumexp are retained for backward.
    Both passes use at most TILE_SIZE x TILE_SIZE attention intermediates.
    """

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        if q.ndim < 2 or k.ndim != q.ndim or v.ndim != q.ndim:
            raise ValueError("Q, K, V must have matching ranks of at least two")
        if q.shape[:-2] != k.shape[:-2] or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
            raise ValueError("Q, K, V must have matching batch/head dimensions; K and V must have identical shapes")
        if min(q.shape[-2:]) < 1 or k.shape[-2] < 1:
            raise ValueError("Sequence lengths and head dimension must be positive")
        if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("Q, K, V must have the same device and dtype")
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("Q, K, V must use a floating-point dtype")
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        o = torch.empty(q.shape, dtype=dtype, device=q.device)
        lse = torch.empty(q.shape[:-1], dtype=dtype, device=q.device)
        for i in range(0, q.shape[-2], TILE_SIZE):
            qi = q[..., i:i + TILE_SIZE, :].to(dtype)
            oi = torch.zeros_like(qi)
            m = torch.full(qi.shape[:-1], -torch.inf, dtype=dtype, device=q.device)
            normalizer = torch.zeros_like(m)
            for j in range(0, k.shape[-2], TILE_SIZE):
                kj = k[..., j:j + TILE_SIZE, :].to(dtype)
                vj = v[..., j:j + TILE_SIZE, :].to(dtype)
                scores = _scores(qi, kj, i, j, is_causal)
                new_m = torch.maximum(m, scores.amax(dim=-1))
                alpha = torch.exp(m - new_m)
                p = torch.exp(scores - new_m[..., None])
                normalizer = alpha * normalizer + p.sum(dim=-1)
                oi = alpha[..., None] * oi + p @ vj
                m = new_m
            o[..., i:i + TILE_SIZE, :] = oi / normalizer[..., None]
            lse[..., i:i + TILE_SIZE] = m + torch.log(normalizer)
        o = o.to(q.dtype)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_o):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = _backward(q, k, v, o, grad_o, lse, ctx.is_causal)
        # apply(Q, K, V) and apply(Q, K, V, is_causal) are both supported.
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None)[:len(ctx.needs_input_grad)]
