import torch
import triton
import triton.language as tl

@triton.jit
def linear_expert_fwd_kernel(
    x_ptr, w_ptr, b_ptr, output_ptr,
    token_mask_ptr,
    batch_seq_len, in_features, out_features, num_experts,
    expert_expansion: tl.constexpr, out_dim
):
    idx = tl.program_id(0)
    if idx >= batch_seq_len:
        return

    m = tl.load(token_mask_ptr + idx)
    if m == 0:
        return

    exponent = num_experts - m
    D_m = (in_features >> exponent) if expert_expansion else (out_features >> exponent)
    D_m = max(D_m, 1)

    x_offset = idx * in_features
    x_i = tl.load(
        x_ptr + x_offset + tl.arange(0, D_m),
        mask=tl.arange(0, D_m) < in_features,
        other=0.0
    )

    if expert_expansion:
        w_m = tl.load(
            w_ptr + tl.arange(0, out_dim * D_m),
            mask=tl.arange(0, out_dim * D_m) < out_dim * in_features,
            other=0.0
        ).to(tl.float32).reshape((out_dim, D_m))
        b_m = tl.load(b_ptr + tl.arange(0, out_dim)) if b_ptr != 0 else 0.0
        y_i = tl.dot(w_m, x_i.to(tl.float32)) + b_m
    else:
        w_m = tl.load(
            w_ptr + tl.arange(0, D_m * out_dim),
            mask=tl.arange(0, D_m * out_dim) < in_features * out_dim,
            other=0.0
        ).to(tl.float32).reshape((D_m, out_dim))
        b_m = tl.load(b_ptr + tl.arange(0, D_m)) if b_ptr != 0 else 0.0
        y_i = tl.dot(x_i.to(tl.float32), w_m) + b_m
        padding = out_dim - D_m
        if padding > 0:
            y_i = tl.cat([y_i, tl.zeros((padding,), dtype=y_i.dtype)])

    output_offset = idx * out_dim
    tl.store(
        output_ptr + output_offset + tl.arange(0, out_dim),
        y_i.to(output_ptr.dtype),
        mask=tl.arange(0, out_dim) < out_dim
    )

@triton.jit
def linear_expert_bwd_kernel(
    x_ptr, w_ptr, dy_ptr,
    dx_ptr, dw_ptr, db_ptr,
    token_mask_ptr,
    batch_seq_len, in_features, out_features, num_experts,
    expert_expansion: tl.constexpr, out_dim,
    USE_BIAS: tl.constexpr
):
    idx = tl.program_id(0)
    if idx >= batch_seq_len:
        return

    m = tl.load(token_mask_ptr + idx)
    if m == 0:
        return

    exponent = num_experts - m
    D_m = (in_features >> exponent) if expert_expansion else (out_features >> exponent)
    D_m = max(D_m, 1)

    x_offset = idx * in_features
    x_i = tl.load(
        x_ptr + x_offset + tl.arange(0, D_m),
        mask=tl.arange(0, D_m) < in_features,
        other=0.0
    ).to(tl.float32)

    dy_offset = idx * out_dim
    dy_i = tl.load(
        dy_ptr + dy_offset + tl.arange(0, out_dim),
        mask=tl.arange(0, out_dim) < out_dim,
        other=0.0
    ).to(tl.float32)

    if expert_expansion:
        w_m = tl.load(
            w_ptr + tl.arange(0, out_dim * D_m),
            mask=tl.arange(0, out_dim * D_m) < out_dim * in_features,
            other=0.0
        ).reshape((out_dim, D_m))
        dx_i = tl.dot(w_m.T, dy_i)
    else:
        w_m = tl.load(
            w_ptr + tl.arange(0, D_m * out_dim),
            mask=tl.arange(0, D_m * out_dim) < in_features * out_dim,
            other=0.0
        ).reshape((D_m, out_dim))
        dx_i = tl.dot(dy_i, w_m.T)

    tl.store(
        dx_ptr + x_offset + tl.arange(0, D_m),
        dx_i.to(dx_ptr.dtype),
        mask=tl.arange(0, D_m) < in_features
    )

    if expert_expansion:
        dw_offset = 0
        dw_shape = (out_dim, D_m)
        dw_indices = tl.arange(0, out_dim * D_m)
        dy_i_broadcast = dy_i[:, None]
        x_i_broadcast = x_i[None, :]
        dw_ij = dy_i_broadcast * x_i_broadcast
        dw_ij_flat = dw_ij.reshape(-1)
        for idx_dw in range(out_dim * D_m):
            idx_w = dw_offset + idx_dw
            val = dw_ij_flat[idx_dw]
            tl.atomic_add(dw_ptr + idx_w, val)
    else:
        dw_offset = 0
        dw_shape = (D_m, out_dim)
        dw_indices = tl.arange(0, D_m * out_dim)
        x_i_broadcast = x_i[:, None]
        dy_i_broadcast = dy_i[None, :]
        dw_ij = x_i_broadcast * dy_i_broadcast
        dw_ij_flat = dw_ij.reshape(-1)
        for idx_dw in range(D_m * out_dim):
            idx_w = dw_offset + idx_dw
            val = dw_ij_flat[idx_dw]
            tl.atomic_add(dw_ptr + idx_w, val)

    if USE_BIAS:
        if expert_expansion:
            for idx_b in range(out_dim):
                val = dy_i[idx_b]
                tl.atomic_add(db_ptr + idx_b, val)
        else:
            for idx_b in range(D_m):
                val = dy_i[idx_b]
                tl.atomic_add(db_ptr + idx_b, val)

class LinearExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, token_mask, num_experts, expert_expansion, in_features, out_features):
        """
        x: Input tensor of shape (batch_size, seq_len, in_features)
        w: Weight tensor
        b: Bias tensor or None
        token_mask: Tensor indicating expert assignments of shape (batch_size, seq_len)
        num_experts: Total number of experts
        expert_expansion: Boolean flag indicating expansion or contraction
        in_features: Input feature dimension
        out_features: Output feature dimension
        """
        batch_size, seq_len, _ = x.shape
        batch_seq_len = batch_size * seq_len
        out_dim = out_features

        # Reshape x and token_mask
        x_reshaped = x.view(batch_seq_len, in_features)
        token_mask_flat = token_mask.view(batch_seq_len)

        # Allocate output tensor
        output = torch.empty((batch_seq_len, out_dim), device=x.device, dtype=x.dtype)

        # Launch the forward Triton kernel
        grid = (batch_seq_len,)
        linear_expert_fwd_kernel[grid](
            x_ptr=x_reshaped,
            w_ptr=w,
            b_ptr=b if b is not None else torch.tensor(0, device=x.device, dtype=x.dtype),
            output_ptr=output,
            token_mask_ptr=token_mask_flat,
            batch_seq_len=batch_seq_len,
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
            expert_expansion=expert_expansion,
            out_dim=out_dim
        )

        # Save context for backward
        ctx.save_for_backward(x_reshaped, w, b, token_mask_flat)
        ctx.num_experts = num_experts
        ctx.expert_expansion = expert_expansion
        ctx.in_features = in_features
        ctx.out_features = out_features

        return output.view(batch_size, seq_len, out_dim)

    @staticmethod
    def backward(ctx, grad_output):
        x_reshaped, w, b, token_mask_flat = ctx.saved_tensors
        num_experts = ctx.num_experts
        expert_expansion = ctx.expert_expansion
        in_features = ctx.in_features
        out_features = ctx.out_features

        batch_seq_len = x_reshaped.shape[0]
        out_dim = out_features

        # Allocate gradient tensors
        dx = torch.empty_like(x_reshaped)
        dw = torch.zeros_like(w)
        db = torch.zeros_like(b) if b is not None else None

        # Reshape grad_output
        grad_output_flat = grad_output.view(batch_seq_len, out_dim)

        # Launch the backward Triton kernel
        grid = (batch_seq_len,)
        USE_BIAS = b is not None

        linear_expert_bwd_kernel[grid](
            x_ptr=x_reshaped,
            w_ptr=w,
            dy_ptr=grad_output_flat,
            dx_ptr=dx,
            dw_ptr=dw,
            db_ptr=db if USE_BIAS else torch.tensor(0, device=x_reshaped.device, dtype=x_reshaped.dtype),
            token_mask_ptr=token_mask_flat,
            batch_seq_len=batch_seq_len,
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
            expert_expansion=expert_expansion,
            out_dim=out_dim,
            USE_BIAS=USE_BIAS
        )

        # Reshape dx to match the input shape
        dx = dx.view(-1, ctx.in_features)
        dx = dx.view_as(ctx.saved_tensors[0])

        # Return gradients
        return dx.view(-1, ctx.in_features).view_as(ctx.saved_tensors[0]), dw, db, None, None, None, None, None
    
def linear_expert(x, w, b, token_mask, num_experts, expert_expansion, in_features, out_features):
    return LinearExpertFunction.apply(x, w, b, token_mask, num_experts, expert_expansion, in_features, out_features)