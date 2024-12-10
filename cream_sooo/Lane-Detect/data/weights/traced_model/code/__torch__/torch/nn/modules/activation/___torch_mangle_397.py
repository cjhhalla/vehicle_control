class LeakyReLU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.activation.___torch_mangle_397.LeakyReLU,
    argument_1: Tensor) -> Tensor:
    input = torch.leaky_relu_(argument_1, 0.10000000000000001)
    return input
