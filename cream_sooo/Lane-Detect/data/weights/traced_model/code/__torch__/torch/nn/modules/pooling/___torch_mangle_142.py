class MaxPool2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.pooling.___torch_mangle_142.MaxPool2d,
    argument_1: Tensor) -> Tensor:
    _0 = torch.max_pool2d(argument_1, [5, 5], [1, 1], [2, 2], [1, 1], False)
    return _0
