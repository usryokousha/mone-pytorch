import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable


class NestedLinearExpand(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_experts=4,
        device=None,
        dtype=None,
        sparse: bool = False,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        return nested_linear_expand(
            x, self.weight, expert_mask, self.bias, self.num_experts
        )


class NestedLinearContract(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_experts=4,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        return nested_linear_contract(
            x, self.weight, expert_mask, self.bias, self.num_experts
        )


def nested_linear_expand(
    x: torch.Tensor,
    w: torch.Tensor,
    expert_mask: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    num_experts: int = 4,
) -> torch.Tensor:
    in_dim = w.shape[1]
    out_dim = w.shape[0]
    input_shape = x.shape
    batch_seq = x.shape[:-1].numel()

    with torch.amp.autocast(device_type="cuda", dtype=torch.get_autocast_gpu_dtype()):
        output = torch.zeros(
            (batch_seq, out_dim), device=x.device, dtype=torch.get_autocast_gpu_dtype()
        )
    x = x.reshape(batch_seq, in_dim)
    for m in range(num_experts):
        # get the valid mask for the m-th expert
        valid_mask = (expert_mask == m).view(batch_seq)

        D_m = in_dim >> (num_experts - m - 1)

        # slice the input and weight
        x_m = x[valid_mask, :D_m]
        w_m = w[:, :D_m]

        # project up to the expert dim
        output[valid_mask, :] = F.linear(x_m, w_m, b)

    return output.reshape(input_shape[:-1] + (out_dim,))


def nested_linear_contract(
    x: torch.Tensor,
    w: torch.Tensor,
    expert_mask: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    num_experts: int = 4,
) -> torch.Tensor:
    in_dim = w.shape[1]
    out_dim = w.shape[0]
    input_shape = x.shape
    batch_seq = x.shape[:-1].numel()

    with torch.amp.autocast(device_type="cuda", dtype=torch.get_autocast_gpu_dtype()):
        output = torch.zeros(
            (batch_seq, out_dim), device=x.device, dtype=torch.get_autocast_gpu_dtype()
        )
    x = x.reshape(batch_seq, in_dim)
    for m in range(num_experts):
        # get the valid mask for the m-th expert
        valid_mask = (expert_mask == m).view(batch_seq)

        D_m = out_dim >> (num_experts - m - 1)

        # get the m-th expert's input and sliced weight
        x_m = x[valid_mask]
        w_m = w[:D_m, :]

        # project down to the expert dim
        if b is not None:
            b_m = b[:D_m]
        else:
            b_m = None

        # Avoid explicit padding
        y = F.linear(x_m, w_m, b_m)
        output[valid_mask, :D_m] = y

    return output.reshape(input_shape[:-1] + (out_dim,))


def nested_linear_expand_sparse_coo(
    x: torch.Tensor,
    w: torch.Tensor,
    expert_mask: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    num_experts: int = 4,
) -> torch.Tensor:
    in_dim = w.shape[1]
    out_dim = w.shape[0]  # Maximum possible D_m
    input_shape = x.shape
    batch_seq = x.shape[:-1].numel()

    x = x.reshape(batch_seq, in_dim)
    expert_mask = expert_mask.view(batch_seq)

    # Compute D_m for each sample based on the expert_mask
    D_m = out_dim >> (num_experts - expert_mask - 1)  # Shape: [batch_seq]

    # Use in_dim directly since D_m_max = in_dim
    indices_j_full = (
        torch.arange(in_dim, device=x.device).unsqueeze(0).expand(batch_seq, out_dim)
    )
    valid_mask = indices_j_full < D_m.unsqueeze(1)  # Shape: [batch_seq, in_dim]

    # Flatten valid indices
    indices_i = (
        torch.arange(batch_seq, device=x.device).unsqueeze(1).expand_as(indices_j_full)
    )
    indices_i = indices_i[valid_mask]  # Shape: [total_nnz]
    indices_j = indices_j_full[valid_mask]  # Shape: [total_nnz]

    # Gather values from x
    values = x[indices_i, indices_j]  # Shape: [total_nnz]

    # Create sparse tensor
    indices = torch.stack([indices_i, indices_j])
    x_sparse = torch.sparse_coo_tensor(indices, values, size=(batch_seq, out_dim))

    # Use full weight matrix w
    w_t = w.t()  # Shape: [in_dim, out_dim]

    # Prepare bias matrix to include in addmm
    if b is not None:
        # Instead of expanding b to [batch_seq, out_dim], we can use a workaround
        # Create an empty output tensor and set its rows to the bias
        bias_output = b.unsqueeze(0).expand(batch_seq, -1).clone()
    else:
        bias_output = torch.zeros(batch_seq, out_dim, device=x.device, dtype=x.dtype)

    # Perform sparse matrix multiplication with bias included
    output = torch.sparse.addmm(
        input=bias_output,
        mat1=x_sparse,
        mat2=w_t,
        beta=1,
        alpha=1,
    )

    return output.reshape(input_shape[:-1] + (out_dim,))


def nested_linear_expand_sparse(
    x: torch.Tensor,
    w: torch.Tensor,
    expert_mask: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    num_experts: int = 4,
) -> torch.Tensor:
    # Ensure that x, w, and b are on the same device and have the correct dtype
    device = x.device
    dtype = torch.float16 if x.dtype == torch.float16 else torch.float32

    x = x.to(dtype)
    w = w.to(dtype)
    if b is not None:
        b = b.to(dtype)

    in_dim = w.shape[1]  # Maximum possible D_m
    out_dim = w.shape[0]
    input_shape = x.shape
    batch_seq = x.shape[:-1].numel()

    x = x.reshape(batch_seq, in_dim)
    expert_mask = expert_mask.view(batch_seq)

    # Compute D_m for each sample based on the expert_mask
    D_m = out_dim >> (num_experts - expert_mask - 1)  # Shape: [batch_seq]

    # Use in_dim directly since D_m_max = in_dim
    indices_j_full = (
        torch.arange(out_dim, device=device).unsqueeze(0).expand(batch_seq, out_dim)
    )
    valid_mask = indices_j_full < D_m.unsqueeze(1)  # Shape: [batch_seq, out_dim]

    # Compute the number of non-zero elements per row
    nnz_per_row = valid_mask.sum(dim=1)

    # Compute crow_indices
    crow_indices = torch.zeros(batch_seq + 1, dtype=torch.int64, device=device)
    torch.cumsum(nnz_per_row, dim=0, out=crow_indices[1:])

    # Get the column indices and values for the non-zero elements
    col_indices = indices_j_full[valid_mask].to(torch.int64)
    values = x[valid_mask]

    # Create the CSR tensor
    x_csr = torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        values,
        size=(batch_seq, out_dim),
        dtype=dtype,
        device=device,
    )

    # Use full weight matrix w
    w_t = w.t()  # Shape: [out_dim, in_dim]

    if b is not None:
        bias_output = b.unsqueeze(0).expand(batch_seq, -1).clone()
    else:
        bias_output = torch.zeros(batch_seq, out_dim, device=device, dtype=dtype)

    # Perform sparse matrix multiplication
    # Note: As of PyTorch 1.12+, torch.sparse.mm supports CSR format
    output = torch.sparse.addmm(
        input=bias_output, mat1=x_csr, mat2=w_t, beta=1, alpha=1
    )

    # Reshape output to match input shape
    output = output.reshape(input_shape[:-1] + (out_dim,))

    return output


def nested_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_mask: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: Callable = F.gelu,
    drop_rate: float = 0.0,
    num_experts: int = 4,
    training: bool = True,
) -> torch.Tensor:
    """More efficient implementation of nested MLP"""
    in_dim = w1.shape[1]
    out_dim = w2.shape[0]
    input_shape = x.shape
    batch_seq = x.shape[:-1].numel()

    with torch.amp.autocast(device_type="cuda", dtype=torch.get_autocast_gpu_dtype()):
        output = torch.zeros(
            (batch_seq, out_dim), device=x.device, dtype=torch.get_autocast_gpu_dtype()
        )
        x = x.reshape(batch_seq, in_dim)
        for m in range(num_experts):
            valid_mask = (expert_mask == m).view(batch_seq)
            D_m_in = in_dim >> (num_experts - m - 1)
            D_m_out = out_dim >> (num_experts - m - 1)

            x_m = x[valid_mask, :D_m_in]
            w_m = w1[:, :D_m_in]
            x_m = F.linear(x_m, w_m, b1)
            x_m = activation(x_m)
            x_m = F.dropout(x_m, drop_rate, training)
            w_m = w2[:D_m_out, :]
            y_m = F.linear(x_m, w_m, b2[:D_m_out])
            output[valid_mask, :D_m_out] = F.dropout(y_m, drop_rate, training)

    return output.reshape(input_shape[:-1] + (out_dim,))

def nested_mlp_sparse(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_mask: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: Callable = F.gelu,
    drop_rate: float = 0.0,
    num_experts: int = 4,
    training: bool = True,
) -> torch.Tensor:
    output = nested_linear_expand_sparse(x, w1, expert_mask, b1, num_experts)
    output = activation(output)
    output = F.dropout(output, drop_rate, training)
    output = nested_linear_contract(output, w2, expert_mask, b2, num_experts)
    return output


def nested_swiglu_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_mask: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    num_experts: int = 4,
) -> torch.Tensor:
    """More efficient implementation of nested MLP with SwiGLU"""
    in_dim = w1.shape[1]
    out_dim = w2.shape[0]
    input_shape = x.shape
    batch_seq = x.shape[:-1].numel()

    with torch.amp.autocast(device_type="cuda", dtype=torch.get_autocast_gpu_dtype()):
        output = torch.zeros(
            (batch_seq, out_dim), device=x.device, dtype=torch.get_autocast_gpu_dtype()
        )
    x = x.reshape(batch_seq, in_dim)
    for m in range(num_experts):
        valid_mask = (expert_mask == m).view(batch_seq)

        D_m_in = in_dim >> (num_experts - m - 1)
        D_m_out = out_dim >> (num_experts - m - 1)

        x_m = x[valid_mask, :D_m_in]
        w_m = w1[:, :D_m_in]

        x_m = F.linear(x_m, w_m, b1)
        x1, x2 = x_m.chunk(2, dim=-1)
        x_m = x1 * F.silu(x2)

        w_m = w2[:D_m_out, :]
        y = F.linear(x_m, w_m, b2[:D_m_out])
        output[valid_mask, :D_m_out] = y

    return output.reshape(input_shape[:-1] + (out_dim,))


if __name__ == "__main__":
    import os

    os.environ["TORCH_CUDNN_V9_API_ENABLED"] = "1"
    x = torch.randn(10, 768, dtype=torch.float32).cuda()
    w = torch.randn(768, 768, dtype=torch.float32).cuda()
    expert_mask = torch.randint(0, 4, (10,)).cuda()
    print(nested_linear_expand_sparse_coo(x, w, expert_mask))
    print(nested_linear_expand_sparse(x, w, expert_mask))
    print(nested_linear_expand(x, w, expert_mask))
    # let's compare latency
    import time

    # Average over 10 executions for dense
    dense_times = []
    for _ in range(10):
        start_time = time.time()
        nested_linear_expand(x, w, expert_mask)
        torch.cuda.synchronize()
        end_time = time.time()
        dense_times.append(end_time - start_time)
    print(f"Dense average time: {sum(dense_times)/10:.6f} seconds")

    # Average over 10 executions for sparse
    sparse_times = []
    for _ in range(10):
        start_time = time.time()
        nested_linear_expand_sparse(x, w, expert_mask)
        torch.cuda.synchronize()
        end_time = time.time()
        sparse_times.append(end_time - start_time)
    print(f"Sparse average time: {sum(sparse_times)/10:.6f} seconds")

    sparse_csr_times = []
    for _ in range(10):
        start_time = time.time()
        nested_linear_expand_sparse(x, w, expert_mask)
        torch.cuda.synchronize()
        end_time = time.time()
        sparse_times.append(end_time - start_time)
    print(f"Sparse CSR average time: {sum(sparse_csr_times)/10:.6f} seconds")

    # Compare half precision nested linear expand with full precision nested linear expand sparse
    x = x.to(torch.bfloat16)
    w = w.to(torch.bfloat16)
    dense_times = []
    for _ in range(10):
        start_time = time.time()
        nested_linear_expand(x, w, expert_mask)
        torch.cuda.synchronize()
        end_time = time.time()
        dense_times.append(end_time - start_time)
    print(f"Half dense average time: {sum(dense_times)/10:.6f} seconds")

    sparse_csr_times = []
    for _ in range(10):
        start_time = time.time()
        nested_linear_expand_sparse(x, w, expert_mask)
        torch.cuda.synchronize()
        end_time = time.time()
        sparse_times.append(end_time - start_time)
    print(f"Half sparse average time: {sum(sparse_times)/10:.6f} seconds")
