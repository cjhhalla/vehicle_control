class RepConv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  act : __torch__.torch.nn.modules.activation.___torch_mangle_286.SiLU
  rbr_reparam : __torch__.torch.nn.modules.conv.___torch_mangle_287.Conv2d
  def forward(self: __torch__.models.common.___torch_mangle_288.RepConv,
    argument_1: Tensor) -> Tensor:
    _0 = self.act
    _1 = (self.rbr_reparam).forward(argument_1, )
    return (_0).forward(_1, )
