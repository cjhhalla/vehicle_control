class BatchNorm2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = ["running_mean", "running_var", "num_batches_tracked", ]
  weight : Tensor
  bias : Tensor
  running_mean : Tensor
  running_var : Tensor
  num_batches_tracked : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.batchnorm.BatchNorm2d,
    input: Tensor) -> Tensor:
    _0 = self.running_var
    _1 = self.running_mean
    _2 = self.bias
    input0 = torch.batch_norm(input, self.weight, _2, _1, _0, False, 0.029999999999999999, 0.001, True)
    return input0
