class Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_74.Conv2d
  act : __torch__.torch.nn.modules.activation.___torch_mangle_75.SiLU
  def forward(self: __torch__.models.common.___torch_mangle_76.Conv,
    argument_1: Tensor) -> Tensor:
    _0 = (self.act).forward((self.conv).forward(argument_1, ), )
    return _0