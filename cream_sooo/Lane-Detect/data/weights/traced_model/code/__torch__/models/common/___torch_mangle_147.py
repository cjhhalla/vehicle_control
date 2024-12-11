class Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_145.Conv2d
  act : __torch__.torch.nn.modules.activation.___torch_mangle_146.SiLU
  def forward(self: __torch__.models.common.___torch_mangle_147.Conv,
    input: Tensor) -> Tensor:
    _0 = (self.act).forward((self.conv).forward(input, ), )
    return _0
