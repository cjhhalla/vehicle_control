class Conv2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.conv.Conv2d,
    input: Tensor) -> Tensor:
    _0 = self.bias
    input0 = torch._convolution(input, self.weight, _0, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    return input0
class ConvTranspose2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.conv.ConvTranspose2d,
    argument_1: Tensor) -> Tensor:
    _1 = self.bias
    input = torch._convolution(argument_1, self.weight, _1, [2, 2], [0, 0], [1, 1], True, [0, 0], 1, False, False, True, True)
    return input
