class ConvTran(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  conv_deconv : __torch__.torch.nn.modules.conv.___torch_mangle_407.ConvTranspose2d
  bn : __torch__.torch.nn.modules.batchnorm.___torch_mangle_408.BatchNorm2d
  act : __torch__.torch.nn.modules.activation.___torch_mangle_409.SiLU
  def forward(self: __torch__.models.common.___torch_mangle_410.ConvTran,
    argument_1: Tensor) -> Tensor:
    _0 = self.act
    _1 = self.bn
    _2 = (self.conv_deconv).forward(argument_1, )
    return (_0).forward((_1).forward(_2, ), )
