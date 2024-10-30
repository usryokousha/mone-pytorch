import torch
import triton
import triton.language as tl

@triton.jit
def get_1d_offset(size, n_prev_chunks):
    return n_prev_chunks * size + tl.arange(0, size)

@triton.jit
def get_2d_offset(offs_0, offs_1, stride_0, stride_1=1):
    return tl.expand_dims(offs_0, 1) * stride_0 + tl.expand_dims(offs_1, 0) * stride_1

@triton.jit
def get_1d_mask(offs, max):
    return offs < max

@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1):
    return (tl.expand_dims(offs_0, 1) < max_0 ) & (tl.expand_dims(offs_1, 0) < max_1)

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

@triton.autotune(
        configs=get_cuda_autotune_config(),
        key=['M', 'N', 'K']
)
@triton.jit
def nested_linear_expand_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    b_ptr,
    mask_ptr,
    M,
    N,
    K,
    E,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    stride_bias,
    stride_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

    # get swizzled program ids
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # get offsets for each axis
    rm = get_1d_offset(BLOCK_SIZE_M, pid_m)
    rn = get_1d_offset(BLOCK_SIZE_N, pid_n)
    rk = get_1d_offset(BLOCK_SIZE_K, 0) # K will always start from 0

    # token mask only operates on the m dimension
    tmask = tl.load(mask_ptr + rm * stride_mask)
    tmask = tl.expand_dims(tmask, 1)

    # initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for e_i in range(E): # e is the number of experts
        # Get in_dim // 2^(e - e_i)
        k_i = K >> (E - e_i - 1)

        # relevant offsets for a and b
        # needs to be updated for each expert
        offs_x = x_ptr + get_2d_offset(rm, rk, stride_xm, stride_xk)
        offs_w = w_ptr + get_2d_offset(rk, rn, stride_wk, stride_wn)
        
        # get masks for the current expert
        t_mask_i = (tmask == (e_i))
        x_mask_i = get_2d_mask(rm, rk, M, k_i)
        w_mask_i = get_2d_mask(rk, rn, k_i, N)

        # perform dot product for the current expert dimension
        for _ in range(0, k_i, BLOCK_SIZE_K):
            x = tl.load(offs_x, mask=t_mask_i & x_mask_i, other=0.0)
            w = tl.load(offs_w, mask=w_mask_i, other=0.0)
            acc += tl.dot(x, w, allow_tf32=True)

            # update offsets for next iteration
            offs_x += BLOCK_SIZE_K * stride_xk
            offs_w += BLOCK_SIZE_K * stride_wk

    # Add bias after accumulation
    if b_ptr is not None:
        offs_b = b_ptr + rn * stride_bias
        b_mask = rn < N
        b = tl.load(offs_b, mask=b_mask, other=0.0)
        acc += b[:, None]

    o = o_ptr + get_2d_offset(rm, rn, stride_om, stride_on)
    mask = get_2d_mask(rm, rn, M, N)
    tl.store(o, acc.to(x.dtype), mask=mask)

@triton.autotune(
        configs=get_cuda_autotune_config(),
        key=['M', 'N', 'K']
)
@triton.jit
def nested_linear_contract_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    b_ptr,
    mask_ptr,
    M,
    N,
    K,
    E,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    stride_bias,
    stride_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

    # get swizzled program ids
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # get offsets for each axis
    rm = get_1d_offset(BLOCK_SIZE_M, pid_m)
    rn = get_1d_offset(BLOCK_SIZE_N, pid_n)
    rk = get_1d_offset(BLOCK_SIZE_K, 0) # K will always start from 0

    # token mask only operates on the m dimension
    tmask = tl.load(mask_ptr + rm * stride_mask)
    tmask = tl.expand_dims(tmask, 1)

    # initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for e_i in range(E): # e is the number of experts
        # Get out_dim // 2^(e - e_i)
        n_i = N >> (E - e_i - 1)

        # relevant offsets for a and b
        # needs to be updated for each expert
        offs_x = x_ptr + get_2d_offset(rm, rk, stride_xm, stride_xk)
        offs_w = w_ptr + get_2d_offset(rk, rn, stride_wk, stride_wn)

        # get masks for the current expert
        t_mask_i = (tmask == (e_i))
        x_mask_i = get_2d_mask(rm, rk, M, K)
        w_mask_i = get_2d_mask(rk, rn, K, n_i)

        for _ in range(0, n_i, BLOCK_SIZE_N):
            x = tl.load(offs_x, mask=t_mask_i & x_mask_i, other=0.0)
            w = tl.load(offs_w, mask=w_mask_i, other=0.0)
            acc += tl.dot(x, w, allow_tf32=True)

            # update offsets for next iteration
            offs_x += BLOCK_SIZE_N * stride_xk
            offs_w += BLOCK_SIZE_N * stride_wk

    # Add bias after accumulation
    if b_ptr is not None:
        offs_b = b_ptr + rn * stride_bias
        b_mask = rn < N
        b = tl.load(offs_b, mask=b_mask, other=0.0)
        acc += b[:, None]

    o = o_ptr + get_2d_offset(rm, rn, stride_om, stride_on)
    mask = get_2d_mask(rm, rn, M, N)
    tl.store(o, acc.to(x.dtype), mask=mask)

def _nested_linear_expand(x, w, mask, b=None, experts=4):
    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix x must be contiguous"
    M, K = x.shape
    K, N = w.shape
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    nested_linear_expand_kernel[grid](
        x, w, output, b, mask, M, N, K, experts,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        output.stride(0), output.stride(1),
        b.stride(0) if b is not None else None,
        mask.stride(0),
    )
    return output

def _nested_linear_contract(x, w, mask, b=None, experts=4):
    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix x must be contiguous"
    M, K = x.shape
    K, N = w.shape
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    nested_linear_contract_kernel[grid](
        x, w, output, b, mask, M, N, K, experts,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        output.stride(0), output.stride(1),
        b.stride(0) if b is not None else None,
        mask.stride(0),
    )
    return output

class NestedLinearExpandFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias, mask, num_experts):
        x = x.view(-1, x.shape[-1])
        w = w.transpose()
        ctx.save_for_backward(x, w, bias, mask)
        ctx.num_experts = num_experts
        return _nested_linear_expand(x, w, mask, bias, num_experts)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w, bias, mask = ctx.saved_tensors
        num_experts = ctx.num_experts
        dw = _nested_linear_expand(x.transpose(), grad_output, mask, bias, num_experts)
        dx = _nested_linear_expand(grad_output, w.transpose(), mask, bias, num_experts)
        db = grad_output.sum(dim=0)
        return dw, dx, db
    
class NestedLinearContractFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias, mask, num_experts):
        x = x.view(-1, x.shape[-1])
        w = w.transpose()
        ctx.save_for_backward(x, w, bias, mask)
        ctx.num_experts = num_experts
        return _nested_linear_contract(x, w, mask, bias, num_experts)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w, bias, mask = ctx.saved_tensors
        num_experts = ctx.num_experts
        dw = _nested_linear_contract(x.transpose(), grad_output, mask, bias, num_experts)
        dx = _nested_linear_contract(grad_output, w.transpose(), mask, bias, num_experts)
        db = grad_output.sum(dim=0)
        return dx, dw, db
    
nested_linear_expand = NestedLinearExpandFunction.apply
nested_linear_contract = NestedLinearContractFunction.apply