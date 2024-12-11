class ImplicitM(Module):
  __parameters__ = ["implicit", ]
  __buffers__ = []
  implicit : Tensor
  training : bool
  def forward(self: __torch__.models.common.___torch_mangle_296.ImplicitM,
    argument_1: Tensor) -> Tensor:
    _0 = torch.mul(self.implicit, argument_1)
    return _0
