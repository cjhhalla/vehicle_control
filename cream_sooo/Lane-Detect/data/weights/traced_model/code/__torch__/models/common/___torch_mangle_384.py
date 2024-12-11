class ConvTran(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  conv_deconv : __torch__.torch.nn.modules.conv.___torch_mangle_381.ConvTranspose2d
  bn : __torch__.torch.nn.modules.batchnorm.___torch_mangle_382.BatchNorm2d
  act : __torch__.torch.nn.modules.activation.___torch_mangle_383.SiLU
  def forward(self: __torch__.models.common.___torch_mangle_384.ConvTran,
    argument_1: Tensor) -> Tensor:
    _0 = self.act
    _1 = self.bn
    _2 = (self.conv_deconv).forward(argument_1, )
    return (_0).forward((_1).forward(_2, ), )
