class Conv2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_145.Conv2d,
    input: Tensor) -> Tensor:
    _0 = self.bias
    input0 = torch._convolution(input, self.weight, _0, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
    return input0
