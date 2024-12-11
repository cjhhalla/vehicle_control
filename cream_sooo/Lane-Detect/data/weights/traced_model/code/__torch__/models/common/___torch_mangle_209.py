class Concat(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.models.common.___torch_mangle_209.Concat,
    argument_1: Tensor,
    argument_2: Tensor,
    argument_3: Tensor,
    argument_4: Tensor,
    argument_5: Tensor,
    argument_6: Tensor) -> Tensor:
    _0 = [argument_1, argument_2, argument_3, argument_4, argument_5, argument_6]
    return torch.cat(_0, 1)
