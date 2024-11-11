# Code taking heavy inspiration from:
#  https://github.com/gpu-mode/lectures
#  https://github.com/Dao-AILab/flash-attention

from torch.amp import custom_bwd, custom_fwd
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
    return (tl.expand_dims(offs_0, 1) < max_0) & (tl.expand_dims(offs_1, 0) < max_1)


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ]


# Let's add a bias to compute the output of a linear layer
@triton.autotune(configs=get_cuda_autotune_config(), key=["M", "N", "K"])
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
    GROUP_SIZE_M: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

    # get swizzled program ids
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # get offsets for each axis
    rm = get_1d_offset(BLOCK_SIZE_M, pid_m)
    rn = get_1d_offset(BLOCK_SIZE_N, pid_n)
    rk = get_1d_offset(BLOCK_SIZE_K, 0)  # K will always start from 0

    # token mask only operates on the m dimension
    tmask = tl.load(mask_ptr + rm * stride_mask)
    tmask = tl.expand_dims(tmask, 1)

    # initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for e_i in range(E):  # e is the number of experts
        # Get in_dim // 2^(e - e_i)
        k_i = K >> (E - e_i - 1)

        # relevant offsets for a and b
        # needs to be updated for each expert
        offs_x = x_ptr + get_2d_offset(rm, rk, stride_xm, stride_xk)
        offs_w = w_ptr + get_2d_offset(rk, rn, stride_wk, stride_wn)

        # get masks for the current expert
        t_mask_i = tmask == (e_i)
        x_mask_i = get_2d_mask(rm, rk, M, k_i)
        w_mask_i = get_2d_mask(rk, rn, k_i, N)

        # perform dot product for the current expert dimension
        for _ in range(0, k_i, BLOCK_SIZE_K):
            x = tl.load(offs_x, mask=t_mask_i & x_mask_i, other=0.0)
            w = tl.load(offs_w, mask=w_mask_i, other=0.0)
            acc += tl.dot(x, w)

            # update offsets for next iteration
            offs_x += BLOCK_SIZE_K * stride_xk
            offs_w += BLOCK_SIZE_K * stride_wk

    # Add bias after accumulation
    if b_ptr is not None:
        offs_b = b_ptr + rn * stride_bias
        b_mask = rn < N
        b = tl.load(offs_b, mask=b_mask, other=0.0)
        acc += b[None, :]

    o = o_ptr + get_2d_offset(rm, rn, stride_om, stride_on)
    mask = get_2d_mask(rm, rn, M, N)
    tl.store(o, acc, mask=mask)


def nested_linear_expand(x, w, mask, b=None, experts=4):
    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix x must be contiguous"
    M, K = x.shape
    K, N = w.shape
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    nested_linear_expand_kernel[grid](
        x,
        w,
        output,
        b,
        mask,
        M,
        N,
        K,
        experts,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        output.stride(0),
        output.stride(1),
        b.stride(0) if b is not None else None,
        mask.stride(0),
    )
    return output


# Let's build the nested linear contraction
@triton.autotune(configs=get_cuda_autotune_config(), key=["M", "N", "K"])
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
    GROUP_SIZE_M: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

    # get swizzled program ids
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # get offsets for each axis
    rm = get_1d_offset(BLOCK_SIZE_M, pid_m)
    rn = get_1d_offset(BLOCK_SIZE_N, pid_n)
    rk = get_1d_offset(BLOCK_SIZE_K, 0)  # K will always start from 0

    # token mask only operates on the m dimension
    tmask = tl.load(mask_ptr + rm * stride_mask)
    tmask = tl.expand_dims(tmask, 1)

    # mask for the current expert
    x_mask_i = get_2d_mask(rm, rk, M, K)

    # initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for e_i in range(E):  # e is the number of experts
        # Get out_dim // 2^(e - e_i)
        n_i = N >> (E - e_i - 1)

        # relevant offsets for a and b
        # needs to be updated for each expert
        offs_x = x_ptr + get_2d_offset(rm, rk, stride_xm, stride_xk)
        offs_w = w_ptr + get_2d_offset(rk, rn, stride_wk, stride_wn)

        # get masks for the current expert
        t_mask_i = tmask == (e_i)
        w_mask_i = get_2d_mask(rk, rn, K, n_i)

        for _ in range(0, K, BLOCK_SIZE_K):
            x = tl.load(offs_x, mask=t_mask_i & x_mask_i, other=0.0)
            w = tl.load(offs_w, mask=w_mask_i, other=0.0)
            acc += tl.dot(x, w)

            # update offsets for next iteration
            offs_x += BLOCK_SIZE_K * stride_xk
            offs_w += BLOCK_SIZE_K * stride_wk

    # Add bias after accumulation
    if b_ptr is not None:
        offs_b = b_ptr + rn * stride_bias
        b_mask = rn < N
        b = tl.load(offs_b, mask=b_mask, other=0.0)
        acc += b[None, :]

    o = o_ptr + get_2d_offset(rm, rn, stride_om, stride_on)
    mask = get_2d_mask(rm, rn, M, N)
    tl.store(o, acc, mask=mask)


def nested_linear_contract(x, w, mask, b=None, experts=4):
    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix x must be contiguous"
    M, K = x.shape
    K, N = w.shape
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    nested_linear_contract_kernel[grid](
        x,
        w,
        output,
        b,
        mask,
        M,
        N,
        K,
        experts,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        output.stride(0),
        output.stride(1),
        b.stride(0) if b is not None else None,
        mask.stride(0),
    )
    return output


# Let's add a bias to compute the output of a linear layer
@triton.autotune(configs=get_cuda_autotune_config(), key=["M", "N", "K"])
@triton.jit
def nested_linear_expand_dx_kernel(
    dy_ptr,
    wT_ptr,
    dx_ptr,
    mask_ptr,
    M,
    N,
    K,
    E,
    stride_dym,
    stride_dyk,
    stride_wTk,
    stride_wTn,
    stride_dxm,
    stride_dxn,
    stride_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

    # get swizzled program ids
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # get offsets for each axis
    rm = get_1d_offset(BLOCK_SIZE_M, pid_m)
    rn = get_1d_offset(BLOCK_SIZE_N, pid_n)
    rk = get_1d_offset(BLOCK_SIZE_K, 0)  # K will always start from 0

    # token mask only operates on the m dimension
    tmask = tl.load(mask_ptr + rm * stride_mask)
    tmask = tl.expand_dims(tmask, 1)

    # initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for e_i in range(E):  # e is the number of experts
        # Get in_dim // 2^(e - e_i)
        n_i = K >> (E - e_i - 1)

        # relevant offsets for a and b
        # needs to be updated for each expert
        offs_dy = dy_ptr + get_2d_offset(rm, rk, stride_dym, stride_dyk)
        offs_wT = wT_ptr + get_2d_offset(rk, rn, stride_wTk, stride_wTn)

        # get masks for the current expert
        t_mask_i = tmask == (e_i)
        dy_mask_i = get_2d_mask(rm, rk, M, K)
        wT_mask_i = get_2d_mask(rk, rn, K, n_i)

        # perform dot product for the current expert dimension
        for _ in range(0, K, BLOCK_SIZE_K):
            dy = tl.load(offs_dy, mask=t_mask_i & dy_mask_i, other=0.0)
            wT = tl.load(offs_wT, mask=wT_mask_i, other=0.0)
            acc += tl.dot(dy, wT)

            # update offsets for next iteration
            offs_dy += BLOCK_SIZE_K * stride_dyk
            offs_wT += BLOCK_SIZE_K * stride_wTn

    dx = dx_ptr + get_2d_offset(rm, rn, stride_dxm, stride_dxn)
    mask = get_2d_mask(rm, rn, M, N)
    tl.store(dx, acc, mask=mask)


def nested_linear_expand_dx(dy, wT, mask, experts=4):
    assert dy.shape[1] == wT.shape[0], "Incompatible dimensions"
    assert dy.is_contiguous(), "Matrix dy must be contiguous"
    M, K = dy.shape
    K, N = wT.shape
    # Allocate output
    dx = torch.empty((M, N), device=dy.device, dtype=dy.dtype)
    # 1D launch kernel where each block gets its own program
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    nested_linear_expand_dx_kernel[grid](
        dy,
        wT,
        dx,
        mask,
        M,
        N,
        K,
        experts,
        dy.stride(0),
        dy.stride(1),
        wT.stride(0),
        wT.stride(1),
        dx.stride(0),
        dx.stride(1),
        mask.stride(0),
    )
    return dx


# Let's add a bias to compute the output of a linear layer
@triton.autotune(configs=get_cuda_autotune_config(), key=["M", "N", "K"])
@triton.jit
def nested_linear_expand_dw_kernel(
    dyT_ptr,
    x_ptr,
    dw_ptr,
    mask_ptr,
    dbias_ptr,
    M,
    N,
    K,
    E,
    stride_dyTm,
    stride_dyTk,
    stride_xk,
    stride_xn,
    stride_dwm,
    stride_dwn,
    stride_mask,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

    # get swizzled program ids
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # get offsets for each axis
    rm = get_1d_offset(BLOCK_SIZE_M, pid_m)
    rn = get_1d_offset(BLOCK_SIZE_N, pid_n)
    rk = get_1d_offset(BLOCK_SIZE_K, 0)  # K will always start from 0

    # token mask only operates on the k dimension
    tmask = tl.load(mask_ptr + rk * stride_mask)

    # initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_bias = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for e_i in range(E):  # e is the number of experts
        # Get out_dim // 2^(e - e_i)
        n_i = N >> (E - e_i - 1)

        # relevant offsets for a and b
        # needs to be updated for each expert
        offs_dyT = dyT_ptr + get_2d_offset(rm, rk, stride_dyTm, stride_dyTk)
        offs_x = x_ptr + get_2d_offset(rk, rn, stride_xk, stride_xn)

        # get masks for the current expert
        t_mask_i = tmask == (e_i)
        dyT_mask_i = get_2d_mask(rm, rk, M, K)
        x_mask_i = get_2d_mask(rk, rn, K, n_i)

        # perform dot product for the current expert dimension
        for _ in range(0, K, BLOCK_SIZE_K):
            dyT = tl.load(
                offs_dyT, mask=tl.expand_dims(t_mask_i, 0) & dyT_mask_i, other=0.0
            )
            x = tl.load(offs_x, mask=tl.expand_dims(t_mask_i, 1) & x_mask_i, other=0.0)
            acc += tl.dot(dyT, x)
            if dbias_ptr is not None:
                acc_bias += tl.sum(dyT, axis=1)

            # update offsets for next iteration
            offs_dyT += BLOCK_SIZE_K * stride_dyTk
            offs_x += BLOCK_SIZE_K * stride_xn

    if dbias_ptr is not None:
        offs_dbias = dbias_ptr + rm * stride_bias
        dbias_mask = rm < M
        tl.store(offs_dbias, acc_bias, mask=dbias_mask)

    dw = dw_ptr + get_2d_offset(rm, rn, stride_dwm, stride_dwn)
    mask = get_2d_mask(rm, rn, M, N)
    tl.store(dw, acc, mask=mask)


def nested_linear_expand_dw(dyT, x, mask, bias=False, experts=4):
    assert dyT.shape[1] == x.shape[0], "Incompatible dimensions"
    assert dyT.is_contiguous(), "Matrix dyT must be contiguous"
    M, K = dyT.shape
    K, N = x.shape
    # Allocate output
    dw = torch.empty((M, N), device=dyT.device, dtype=dyT.dtype)
    dbias = None
    if bias:
        dbias = torch.empty((M,), device=dyT.device, dtype=dyT.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    nested_linear_expand_dw_kernel[grid](
        dyT,
        x,
        dw,
        mask,
        dbias,
        M,
        N,
        K,
        experts,
        dyT.stride(0),
        dyT.stride(1),
        x.stride(0),
        x.stride(1),
        dw.stride(0),
        dw.stride(1),
        mask.stride(0),
        dbias.stride(0) if bias else None,
    )
    return dw, dbias


# Let's add a bias to compute the output of a linear layer
@triton.autotune(configs=get_cuda_autotune_config(), key=["M", "N", "K"])
@triton.jit
def nested_linear_contract_dx_kernel(
    dy_ptr,
    wT_ptr,
    dx_ptr,
    mask_ptr,
    M,
    N,
    K,
    E,
    stride_dym,
    stride_dyk,
    stride_wTk,
    stride_wTn,
    stride_dxm,
    stride_dxn,
    stride_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

    # get swizzled program ids
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # get offsets for each axis
    rm = get_1d_offset(BLOCK_SIZE_M, pid_m)
    rn = get_1d_offset(BLOCK_SIZE_N, pid_n)
    rk = get_1d_offset(BLOCK_SIZE_K, 0)  # K will always start from 0

    # token mask only operates on the m dimension
    tmask = tl.load(mask_ptr + rm * stride_mask)
    tmask = tl.expand_dims(tmask, 1)

    # initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for e_i in range(E):  # e is the number of experts
        # Get in_dim // 2^(e - e_i)
        k_i = K >> (E - e_i - 1)

        # relevant offsets for a and b
        # needs to be updated for each expert
        offs_dy = dy_ptr + get_2d_offset(rm, rk, stride_dym, stride_dyk)
        offs_wT = wT_ptr + get_2d_offset(rk, rn, stride_wTk, stride_wTn)

        # get masks for the current expert
        t_mask_i = tmask == (e_i)
        dy_mask_i = get_2d_mask(rm, rk, M, k_i)
        wT_mask_i = get_2d_mask(rk, rn, k_i, N)

        # perform dot product for the current expert dimension
        for _ in range(0, k_i, BLOCK_SIZE_K):
            dy = tl.load(offs_dy, mask=t_mask_i & dy_mask_i, other=0.0)
            wT = tl.load(offs_wT, mask=wT_mask_i, other=0.0)
            acc += tl.dot(dy, wT)

            # update offsets for next iteration
            offs_dy += BLOCK_SIZE_K * stride_dyk
            offs_wT += BLOCK_SIZE_K * stride_wTn

    dx = dx_ptr + get_2d_offset(rm, rn, stride_dxm, stride_dxn)
    mask = get_2d_mask(rm, rn, M, N)
    tl.store(dx, acc, mask=mask)


def nested_linear_contract_dx(dy, wT, mask, experts=4):
    assert dy.shape[1] == wT.shape[0], "Incompatible dimensions"
    assert dy.is_contiguous(), "Matrix dy must be contiguous"
    M, N = dy.shape
    K, N = wT.shape
    # Allocate output
    dx = torch.empty((M, N), device=dy.device, dtype=dy.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    nested_linear_contract_dx_kernel[grid](
        dy,
        wT,
        dx,
        mask,
        M,
        N,
        K,
        experts,
        dy.stride(0),
        dy.stride(1),
        wT.stride(0),
        wT.stride(1),
        dx.stride(0),
        dx.stride(1),
        mask.stride(0),
    )
    return dx


# Let's add a bias to compute the output of a linear layer
@triton.autotune(configs=get_cuda_autotune_config(), key=["M", "N", "K"])
@triton.jit
def nested_linear_contract_dw_kernel(
    dyT_ptr,
    x_ptr,
    dw_ptr,
    mask_ptr,
    dbias_ptr,
    M,
    N,
    K,
    E,
    stride_dyTm,
    stride_dyTk,
    stride_xk,
    stride_xn,
    stride_dwm,
    stride_dwn,
    stride_mask,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

    # get swizzled program ids
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # get offsets for each axis
    rm = get_1d_offset(BLOCK_SIZE_M, pid_m)
    rn = get_1d_offset(BLOCK_SIZE_N, pid_n)
    rk = get_1d_offset(BLOCK_SIZE_K, 0)  # K will always start from 0

    # token mask only operates on the k dimension
    tmask = tl.load(mask_ptr + rk * stride_mask)

    # initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_bias = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for e_i in range(E):  # e is the number of experts
        # Get out_dim // 2^(e - e_i)
        m_i = M >> (E - e_i - 1)

        # relevant offsets for a and b
        # needs to be updated for each expert
        offs_dyT = dyT_ptr + get_2d_offset(rm, rk, stride_dyTm, stride_dyTk)
        offs_x = x_ptr + get_2d_offset(rk, rn, stride_xk, stride_xn)

        # get masks for the current expert
        t_mask_i = tmask == (e_i)
        dyT_mask_i = get_2d_mask(rm, rk, m_i, K)
        x_mask_i = get_2d_mask(rk, rn, K, N)

        # perform dot product for the current expert dimension
        for _ in range(0, K, BLOCK_SIZE_K):
            dyT = tl.load(
                offs_dyT, mask=tl.expand_dims(t_mask_i, 0) & dyT_mask_i, other=0.0
            )
            x = tl.load(offs_x, mask=tl.expand_dims(t_mask_i, 1) & x_mask_i, other=0.0)
            acc += tl.dot(dyT, x)
            if dbias_ptr is not None:
                acc_bias += tl.sum(dyT, axis=1)

            # update offsets for next iteration
            offs_dyT += BLOCK_SIZE_K * stride_dyTk
            offs_x += BLOCK_SIZE_K * stride_xn

    if dbias_ptr is not None:
        offs_dbias = dbias_ptr + rm * stride_bias
        dbias_mask = rm < M
        tl.store(offs_dbias, acc_bias, mask=dbias_mask)

    dw = dw_ptr + get_2d_offset(rm, rn, stride_dwm, stride_dwn)
    mask = get_2d_mask(rm, rn, M, N)
    tl.store(dw, acc, mask=mask)


def nested_linear_contract_dw(dyT, x, mask, bias=False, experts=4):
    assert dyT.shape[1] == x.shape[0], "Incompatible dimensions"
    assert dyT.is_contiguous(), "Matrix dyT must be contiguous"
    M, N = dyT.shape
    K, N = x.shape
    # Allocate output
    dw = torch.empty((M, N), device=dyT.device, dtype=dyT.dtype)
    dbias = None
    if bias:
        dbias = torch.empty((M,), device=dyT.device, dtype=dyT.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    nested_linear_contract_dw_kernel[grid](
        dyT,
        x,
        dw,
        mask,
        dbias,
        M,
        N,
        K,
        experts,
        dyT.stride(0),
        dyT.stride(1),
        x.stride(0),
        x.stride(1),
        dw.stride(0),
        dw.stride(1),
        mask.stride(0),
        dbias.stride(0) if bias else None,
    )
    return dw, dbias

class NestedLinearExpand(torch.autograd.Function):
    @custom_fwd
    @staticmethod
    def forward(ctx, x, w, mask, bias=None, experts=4):
        input_shape = x.shape
        ctx.input_shape = input_shape
        x = x.reshape(-1, x.shape[-1])
        if x.stride(0) > 1 and x.stride(1) > 1:
            x = x.contiguous()
        if w.stride(0) > 1 and w.stride(1) > 1:
            w = w.contiguous()
        bias = bias.contiguous() if bias is not None else None
        ctx.experts = experts
        ctx.bias = False if bias is None else True
        ctx.save_for_backward(x, w, mask)
        return nested_linear_expand(x, w.transpose(0, 1), mask, bias, experts).view(
            input_shape
        )

    @custom_bwd
    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.input_shape
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        grad_output = grad_output.contiguous()
        x, w, mask = ctx.saved_tensors
        experts = ctx.experts
        bias = ctx.bias
        dx = nested_linear_expand_dx(grad_output, w, mask, experts=experts).view(
            input_shape
        )
        dw, dbias = nested_linear_expand_dw(
            grad_output.transpose(0, 1), x, mask, bias=bias, experts=experts
        )
        return dx, dw, None, dbias, None
    
class NestedLinearContract(torch.autograd.Function):
    @custom_fwd
    @staticmethod
    def forward(ctx, x, w, mask, bias=None, experts=4):
        input_shape = x.shape
        ctx.input_shape = input_shape
        x = x.reshape(-1, x.shape[-1])
        if x.stride(0) > 1 and x.stride(1) > 1:
            x = x.contiguous()
        if w.stride(0) > 1 and w.stride(1) > 1:
            w = w.contiguous()
        bias = bias.contiguous() if bias is not None else None
        ctx.experts = experts
        ctx.bias = False if bias is None else True
        ctx.save_for_backward(x, w, mask)
        return nested_linear_contract(x, w.transpose(0, 1), mask, bias, experts).view(
            input_shape
        )

    @custom_bwd
    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.input_shape
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        grad_output = grad_output.contiguous()
        x, w, mask = ctx.saved_tensors
        experts = ctx.experts
        bias = ctx.bias
        dx = nested_linear_contract_dx(grad_output, w, mask, experts=experts).view(
            input_shape
        )
        dw, dbias = nested_linear_contract_dw(
            grad_output.transpose(0, 1), x, mask, bias=bias, experts=experts
        )
        return dx, dw, None, dbias, None
    
def nested_linear_expand_triton(x, w, mask, bias=None, experts=4):
    return NestedLinearExpand.apply(x, w, mask, bias, experts)

def nested_linear_contract_triton(x, w, mask, bias=None, experts=4):
    return NestedLinearContract.apply(x, w, mask, bias, experts)