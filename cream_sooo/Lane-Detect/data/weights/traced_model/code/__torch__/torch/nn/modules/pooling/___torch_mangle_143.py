class MaxPool2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.pooling.___torch_mangle_143.MaxPool2d,
    argument_1: Tensor) -> Tensor:
    _0 = torch.max_pool2d(argument_1, [9, 9], [1, 1], [4, 4], [1, 1], False)
    return _0
