class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  __annotations__["0"] = __torch__.models.common.___torch_mangle_404.Bottleneck
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_405.Sequential,
    argument_1: Tensor) -> Tensor:
    _0 = (getattr(self, "0")).forward(argument_1, )
    return _0