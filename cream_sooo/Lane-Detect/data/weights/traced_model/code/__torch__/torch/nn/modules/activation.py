class SiLU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.activation.SiLU,
    argument_1: Tensor) -> Tensor:
    return torch.silu(argument_1)
class LeakyReLU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.activation.LeakyReLU,
    argument_1: Tensor) -> Tensor:
    input = torch.leaky_relu_(argument_1, 0.10000000000000001)
    return input
class Sigmoid(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.activation.Sigmoid,
    argument_1: Tensor) -> Tensor:
    return torch.sigmoid(argument_1)
