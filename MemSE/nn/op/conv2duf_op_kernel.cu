
#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>


#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
static __global__ void
conv2duf_weird_term(scalar_t *input, const scalar_t *gamma) {
  int xi = blockIdx.x * loop_x * blockDim.x + threadIdx.x;

  scalar_t zero = 0.0;

  for (int loop_idx = 0; loop_idx < loop_x && xi < size_x;
       loop_idx++, xi += blockDim.x) {
    scalar_t x = p_x[xi];

    if (use_bias) {
      x += p_b[(xi / step_b) % size_b];
    }

    scalar_t ref = use_ref ? p_ref[xi] : zero;

    scalar_t y;

    switch (act * 10 + grad) {
    default:
    case 10:
      y = x;
      break;
    case 11:
      y = x;
      break;
    case 12:
      y = 0.0;
      break;

    case 30:
      y = (x > 0.0) ? x : x * alpha;
      break;
    case 31:
      y = (ref > 0.0) ? x : x * alpha;
      break;
    case 32:
      y = 0.0;
      break;
    }

    out[xi] = y * scale;
  }
}

torch::Tensor conv2duf_op(const torch::Tensor &input,
                                const torch::Tensor &gamma) {
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto x = input.contiguous();
  auto g = gamma.contiguous();

  int size_x = x.numel();
  int size_b = b.numel();
  int step_b = 1;

  for (int i = 1 + 1; i < x.dim(); i++) {
    step_b *= x.size(i);
  }

  int loop_x = 4;
  int block_size = 4 * 32;
  int grid_size = (size_x - 1) / (loop_x * block_size) + 1;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x.scalar_type(), "conv2duf_weird_term", [&] {
        conv2duf_weird_term<scalar_t><<<grid_size, block_size, 0, stream>>>(
            x.data_ptr<scalar_t>(), g.data_ptr<scalar_t>());
      });

  return y;
}