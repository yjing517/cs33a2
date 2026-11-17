"""FlashAttention-2 fused Triton forward with the PyTorch recomputation backward."""

import torch
import triton
import triton.language as tl

from cs336_systems.flash_attention import FlashAttention2


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    query_start = query_tile_index * Q_TILE_SIZE
    q_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb, (N_QUERIES, D),
        (stride_qq, stride_qd), (query_start, 0), (Q_TILE_SIZE, D), (1, 0),
    )
    k_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb, (D, N_KEYS),
        (stride_kd, stride_kk), (0, 0), (D, K_TILE_SIZE), (0, 1),
    )
    v_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb, (N_KEYS, D),
        (stride_vk, stride_vd), (0, 0), (K_TILE_SIZE, D), (1, 0),
    )
    q = tl.load(q_ptr, boundary_check=(0,), padding_option="zero")
    rows = query_start + tl.arange(0, Q_TILE_SIZE)
    m = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
    normalizer = tl.zeros((Q_TILE_SIZE,), tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D), tl.float32)
    for key_start in range(0, N_KEYS, K_TILE_SIZE):
        k = tl.load(k_ptr, boundary_check=(1,), padding_option="zero")
        v = tl.load(v_ptr, boundary_check=(0,), padding_option="zero")
        # IEEE dot avoids the default TF32 rounding of float32 inputs on Ampere.
        scores = tl.dot(q, k, input_precision="ieee") * scale
        cols = key_start + tl.arange(0, K_TILE_SIZE)
        if is_causal:
            scores += tl.where(rows[:, None] >= cols[None, :], 0.0, -1.0e6)
        scores = tl.where(cols[None, :] < N_KEYS, scores, -float("inf"))
        new_m = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m[:, None])
        normalizer = alpha * normalizer + tl.sum(p, axis=1)
        acc = tl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None], input_precision="ieee")
        m = new_m
        k_ptr = tl.advance(k_ptr, (0, K_TILE_SIZE))
        v_ptr = tl.advance(v_ptr, (K_TILE_SIZE, 0))
    o_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob, (N_QUERIES, D),
        (stride_oq, stride_od), (query_start, 0), (Q_TILE_SIZE, D), (1, 0),
    )
    tl.store(o_ptr, (acc / normalizer[:, None]).to(O_ptr.dtype.element_ty), boundary_check=(0,))
    tl.store(L_ptr + batch_index * stride_lb + rows * stride_lq, m + tl.log(normalizer), rows < N_QUERIES)


class FlashAttention2Triton(FlashAttention2):
    """CUDA attention on (batch, sequence, head_dim); inherits tiled backward."""

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("Triton attention expects 3D (batch, sequence, head_dim) tensors")
        if q.shape[0] != k.shape[0] or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
            raise ValueError("Q, K, V must match in batch/head dimensions; K and V must have identical shapes")
        if q.device.type != "cuda" or q.device != k.device or q.device != v.device:
            raise ValueError("Q, K, V must be on the same CUDA device")
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32) or q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("Q, K, V must share float16, bfloat16 or float32 dtype")
        if q.shape[-1] not in (16, 32, 64, 128) or min(q.shape[:2]) < 1 or k.shape[1] < 1:
            raise ValueError("Head dimension must be 16, 32, 64 or 128, with positive batch and sequence lengths")
        o = torch.empty_like(q)
        lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
        tile_size = 32
        with torch.cuda.device(q.device):
            flash_fwd_kernel[(triton.cdiv(q.shape[1], tile_size), q.shape[0])](
                q, k, v, o, lse,
                *q.stride(), *k.stride(), *v.stride(), *o.stride(), *lse.stride(),
                q.shape[1], k.shape[1], q.shape[-1] ** -0.5,
                D=q.shape[-1], Q_TILE_SIZE=tile_size, K_TILE_SIZE=tile_size,
                is_causal=is_causal, num_warps=4,
            )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.is_causal = is_causal
        return o
