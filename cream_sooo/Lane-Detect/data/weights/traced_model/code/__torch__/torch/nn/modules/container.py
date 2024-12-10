class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  __annotations__["0"] = __torch__.torch.nn.modules.pooling.___torch_mangle_142.MaxPool2d
  __annotations__["1"] = __torch__.torch.nn.modules.pooling.___torch_mangle_143.MaxPool2d
  __annotations__["2"] = __torch__.torch.nn.modules.pooling.___torch_mangle_144.MaxPool2d
class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  __annotations__["0"] = __torch__.models.common.Bottleneck
  def forward(self: __torch__.torch.nn.modules.container.Sequential,
    argument_1: Tensor) -> Tensor:
    _0 = (getattr(self, "0")).forward(argument_1, )
    return _0
