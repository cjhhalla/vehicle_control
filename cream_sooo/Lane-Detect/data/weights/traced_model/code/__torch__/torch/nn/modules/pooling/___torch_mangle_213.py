class MaxPool2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.pooling.___torch_mangle_213.MaxPool2d,
    argument_1: Tensor) -> Tensor:
    input = torch.max_pool2d(argument_1, [2, 2], [2, 2], [0, 0], [1, 1], False)
    return input
