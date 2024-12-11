class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  cv1 : __torch__.models.common.___torch_mangle_371.Conv
  cv2 : __torch__.models.common.___torch_mangle_374.Conv
  def forward(self: __torch__.models.common.___torch_mangle_375.Bottleneck,
    argument_1: Tensor) -> Tensor:
    _0 = (self.cv2).forward((self.cv1).forward(argument_1, ), )
    return _0
