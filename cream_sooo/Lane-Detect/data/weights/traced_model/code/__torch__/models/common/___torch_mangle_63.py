class MP(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  m : __torch__.torch.nn.modules.pooling.___torch_mangle_62.MaxPool2d
  def forward(self: __torch__.models.common.___torch_mangle_63.MP,
    argument_1: Tensor) -> Tensor:
    return (self.m).forward(argument_1, )
