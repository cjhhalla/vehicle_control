class MaxPool2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.pooling.MaxPool2d,
    argument_1: Tensor) -> Tensor:
    input = torch.max_pool2d(argument_1, [2, 2], [2, 2], [0, 0], [1, 1], False)
    return input
class AdaptiveAvgPool2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.pooling.AdaptiveAvgPool2d,
    argument_1: Tensor) -> Tensor:
    _0 = torch.adaptive_avg_pool2d(argument_1, [1, 1])
    return _0
