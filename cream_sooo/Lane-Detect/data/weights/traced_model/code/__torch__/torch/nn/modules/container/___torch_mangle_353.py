class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  __annotations__["0"] = __torch__.torch.nn.modules.linear.Linear
  __annotations__["1"] = __torch__.torch.nn.modules.activation.___torch_mangle_350.SiLU
  __annotations__["2"] = __torch__.torch.nn.modules.linear.___torch_mangle_351.Linear
  __annotations__["3"] = __torch__.torch.nn.modules.activation.___torch_mangle_352.Sigmoid
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_353.Sequential,
    input: Tensor) -> Tensor:
    _0 = getattr(self, "3")
    _1 = getattr(self, "2")
    _2 = getattr(self, "1")
    _3 = (getattr(self, "0")).forward(input, )
    _4 = (_0).forward((_1).forward((_2).forward(_3, ), ), )
    return _4
