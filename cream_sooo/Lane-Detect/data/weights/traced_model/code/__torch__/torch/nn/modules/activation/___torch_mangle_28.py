class SiLU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.torch.nn.modules.activation.___torch_mangle_28.SiLU,
    argument_1: Tensor) -> Tensor:
    return torch.silu(argument_1)