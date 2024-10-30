# Mixture of Nested Experts (MoNE) - PyTorch Implementation

This repository contains a PyTorch implementation of the **Mixture of Nested Experts (MoNE)** framework, as described in the paper _[Mixture of Nested Experts: Adaptive Processing of Visual Tokens](https://arxiv.org/abs/2407.19985)_. MoNE is designed for efficient visual token processing by dynamically allocating computational resources, reducing inference costs without sacrificing model accuracy.

## Features

- **ExpertPreferredRouter**: Dynamic routing based on token importance, directing tokens to appropriate experts. [Found in `mone_pytorch/routing.py`](mone_pytorch/routing.py)
- **Nested Linear Projections**: Includes `NestedLinearExpand` and `NestedLinearContract`, implementing nested linear projections for flexible token processing. [Located in `mone_pytorch/layers.py`](mone_pytorch/layers.py)
- **Optimized Triton Kernels**: Custom Triton kernels to support efficient nested linear projections. [See `mone_pytorch/ops/nested_linear_triton.py`](mone_pytorch/ops/nested_linear_triton.py)

## Installation

To run the code, clone this repository and install the required dependencies:

```bash
git clone <repository-url>
cd mone-pytorch
pip install -r requirements.txt
```

## Usage

### 1. ExpertPreferredRouter

The `ExpertPreferredRouter` assigns tokens to nested experts based on importance. Located in `mone_pytorch/routing.py`, this router is the core of MoNE’s dynamic token routing.

### 2. NestedLinearExpand and NestedLinearContract

These classes manage nested linear projections to process tokens at varying computational levels. You can find these implementations in `mone_pytorch/layers.py`.

### 3. Triton Kernels for Nested Projections

For enhanced efficiency, the Triton kernels in `mone_pytorch/ops/nested_linear_triton.py` support the nested projections required for MoNE’s architecture.

## Example

Below is a minimal example to demonstrate initializing and using the MoNE framework:

```python
from mone_pytorch.routing import compute_capacity_distribution
from mone_pytorch.block import NestedBlock

# Define capacity distribution parameters
e_c = 0.6  # Effective capacity (between 0 and 1)
E = 3  # Number of experts
delta = 2  # Incentive parameter (>1)
beta = 10  # Entropy regularization parameter (>0)

# Compute capacity distribution
capacity_distribution = compute_capacity_distribution(e_c, E, delta, beta)

# Define router and layers as per model architecture
block = NestedBlock(
    dim=128,
    num_heads=8,
    num_experts=E,
    capacity_distribution=capacity_distribution,
)

...
```

## References

For a detailed overview of MoNE, please refer to the paper: _Mixture of Nested Experts: Adaptive Processing of Visual Tokens_ by Jain et al.

---

Feel free to modify sections, add specific examples, or link the paper directly.

# To Do
- [x] Build MoNE nested linear layers
- [x] Build efficient triton kernels for nested linear layers
- [x] Create transformer block using MoNE components
- [ ] Integrate MoNE into DINOv2
- [ ] Add Hydra (Mamba v2) variant of MoNE mixer
- [ ] Add example notebooks

# Acknowledgements

- [xformers](https://github.com/facebookresearch/xformers) for the memory efficient attention implementation
- [dinov2](https://github.com/facebookresearch/dinov2) for the implementation of the DINOv2 model
- [hydra](https://github.com/goombalab/hydra) for more attention substitute
