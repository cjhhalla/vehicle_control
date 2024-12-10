class ImplicitA(Module):
  __parameters__ = ["implicit", ]
  __buffers__ = []
  implicit : Tensor
  training : bool
  def forward(self: __torch__.models.common.___torch_mangle_294.ImplicitA,
    argument_1: Tensor) -> Tensor:
    input = torch.add(self.implicit, argument_1, alpha=1)
    return input
