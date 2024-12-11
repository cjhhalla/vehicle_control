class ConvTranspose2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_381.ConvTranspose2d,
    argument_1: Tensor) -> Tensor:
    _0 = self.bias
    input = torch._convolution(argument_1, self.weight, _0, [2, 2], [0, 0], [1, 1], True, [0, 0], 1, False, False, True, True)
    return input
