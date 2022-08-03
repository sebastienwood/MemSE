
#include <ATen/ATen.h>
#include <torch/extension.h>

torch::Tensor conv2duf_op(const torch::Tensor &input,
                                const torch::Tensor &gamma);

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor conv2duf(const torch::Tensor &input,
                             const torch::Tensor &gamma) {
  CHECK_INPUT(input);
  CHECK_INPUT(bias);

  at::DeviceGuard guard(input.device());

  return conv2duf_op(input, gamma);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2duf", &conv2duf, "Conv2DUF (CUDA)");
}