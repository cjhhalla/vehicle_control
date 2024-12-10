class Linear(Module):
  __parameters__ = ["weight", ]
  __buffers__ = []
  weight : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.linear.___torch_mangle_351.Linear,
    argument_1: Tensor) -> Tensor:
    input = torch.matmul(argument_1, torch.t(self.weight))
    return input
