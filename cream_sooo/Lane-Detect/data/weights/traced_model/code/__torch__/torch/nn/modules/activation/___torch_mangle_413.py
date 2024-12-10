class Sigmoid(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.activation.___torch_mangle_413.Sigmoid,
    argument_1: Tensor) -> Tensor:
    return torch.sigmoid(argument_1)
