class Concat(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.models.common.___torch_mangle_258.Concat,
    argument_1: Tensor,
    argument_2: Tensor,
    argument_3: Tensor) -> Tensor:
    _0 = [argument_1, argument_2, argument_3]
    return torch.cat(_0, 1)