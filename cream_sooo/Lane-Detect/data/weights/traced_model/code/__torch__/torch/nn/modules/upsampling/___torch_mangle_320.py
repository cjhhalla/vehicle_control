class Upsample(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.upsampling.___torch_mangle_320.Upsample,
    argument_1: Tensor) -> Tensor:
    input = torch.upsample_nearest2d(argument_1, None, [2., 2.])
    return input
