class Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  conv : __torch__.torch.nn.modules.conv.Conv2d
  act : __torch__.torch.nn.modules.activation.SiLU
  def forward(self: __torch__.models.common.Conv,
    input: Tensor) -> Tensor:
    _0 = (self.act).forward((self.conv).forward(input, ), )
    return _0
class Concat(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.models.common.Concat,
    argument_1: Tensor,
    argument_2: Tensor,
    argument_3: Tensor,
    argument_4: Tensor) -> Tensor:
    _1 = [argument_1, argument_2, argument_3, argument_4]
    return torch.cat(_1, 1)
class MP(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  m : __torch__.torch.nn.modules.pooling.MaxPool2d
  def forward(self: __torch__.models.common.MP,
    argument_1: Tensor) -> Tensor:
    return (self.m).forward(argument_1, )
class SPPCSPC(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  cv1 : __torch__.models.common.___torch_mangle_132.Conv
  cv2 : __torch__.models.common.___torch_mangle_135.Conv
  cv3 : __torch__.models.common.___torch_mangle_138.Conv
  cv4 : __torch__.models.common.___torch_mangle_141.Conv
  m : __torch__.torch.nn.modules.container.ModuleList
  cv5 : __torch__.models.common.___torch_mangle_147.Conv
  cv6 : __torch__.models.common.___torch_mangle_150.Conv
  cv7 : __torch__.models.common.___torch_mangle_153.Conv
  def forward(self: __torch__.models.common.SPPCSPC,
    argument_1: Tensor) -> Tensor:
    _2 = self.cv7
    _3 = self.cv2
    _4 = self.cv6
    _5 = self.cv5
    _6 = getattr(self.m, "2")
    _7 = getattr(self.m, "1")
    _8 = getattr(self.m, "0")
    _9 = self.cv4
    _10 = (self.cv3).forward((self.cv1).forward(argument_1, ), )
    _11 = (_9).forward(_10, )
    _12 = [_11, (_8).forward(_11, ), (_7).forward(_11, ), (_6).forward(_11, )]
    input = torch.cat(_12, 1)
    _13 = (_4).forward((_5).forward(input, ), )
    input0 = torch.cat([_13, (_3).forward(argument_1, )], 1)
    return (_2).forward(input0, )
class RepConv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  act : __torch__.torch.nn.modules.activation.___torch_mangle_281.SiLU
  rbr_reparam : __torch__.torch.nn.modules.conv.___torch_mangle_282.Conv2d
  def forward(self: __torch__.models.common.RepConv,
    argument_1: Tensor) -> Tensor:
    _14 = self.act
    _15 = (self.rbr_reparam).forward(argument_1, )
    return (_14).forward(_15, )
class ImplicitA(Module):
  __parameters__ = ["implicit", ]
  __buffers__ = []
  implicit : Tensor
  training : bool
  def forward(self: __torch__.models.common.ImplicitA,
    argument_1: Tensor) -> Tensor:
    input = torch.add(self.implicit, argument_1, alpha=1)
    return input
class ImplicitM(Module):
  __parameters__ = ["implicit", ]
  __buffers__ = []
  implicit : Tensor
  training : bool
  def forward(self: __torch__.models.common.ImplicitM,
    argument_1: Tensor) -> Tensor:
    _16 = torch.mul(self.implicit, argument_1)
    return _16
class BottleneckCSP(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  cv1 : __torch__.models.common.___torch_mangle_305.Conv
  cv2 : __torch__.torch.nn.modules.conv.___torch_mangle_306.Conv2d
  cv3 : __torch__.torch.nn.modules.conv.___torch_mangle_307.Conv2d
  cv4 : __torch__.models.common.___torch_mangle_310.Conv
  bn : __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  act : __torch__.torch.nn.modules.activation.LeakyReLU
  m : __torch__.torch.nn.modules.container.Sequential
  def forward(self: __torch__.models.common.BottleneckCSP,
    argument_1: Tensor) -> Tensor:
    _17 = self.cv4
    _18 = self.act
    _19 = self.bn
    _20 = self.cv2
    _21 = self.cv3
    _22 = (self.m).forward((self.cv1).forward(argument_1, ), )
    _23 = [(_21).forward(_22, ), (_20).forward(argument_1, )]
    input = torch.cat(_23, 1)
    _24 = (_18).forward((_19).forward(input, ), )
    return (_17).forward(_24, )
class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  cv1 : __torch__.models.common.___torch_mangle_313.Conv
  cv2 : __torch__.models.common.___torch_mangle_316.Conv
  def forward(self: __torch__.models.common.Bottleneck,
    argument_1: Tensor) -> Tensor:
    _25 = (self.cv2).forward((self.cv1).forward(argument_1, ), )
    return _25
class out_Conv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_348.Conv2d
  bn : __torch__.torch.nn.modules.batchnorm.___torch_mangle_349.BatchNorm2d
  act : __torch__.torch.nn.modules.activation.Sigmoid
  def forward(self: __torch__.models.common.out_Conv,
    argument_1: Tensor) -> Tensor:
    _26 = self.act
    _27 = (self.bn).forward((self.conv).forward(argument_1, ), )
    return (_26).forward(_27, )
class SE_AT(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  avg_pool : __torch__.torch.nn.modules.pooling.AdaptiveAvgPool2d
  fc : __torch__.torch.nn.modules.container.___torch_mangle_353.Sequential
  c1 : __torch__.models.common.___torch_mangle_356.Conv
  def forward(self: __torch__.models.common.SE_AT,
    argument_1: Tensor) -> Tensor:
    _28 = self.c1
    _29 = self.fc
    _30 = self.avg_pool
    b = ops.prim.NumToTensor(torch.size(argument_1, 0))
    _31 = int(b)
    _32 = int(b)
    c = ops.prim.NumToTensor(torch.size(argument_1, 1))
    _33 = int(c)
    _34 = int(c)
    input = torch.view((_30).forward(argument_1, ), [_32, _34])
    y = torch.view((_29).forward(input, ), [_31, _33, 1, 1])
    y1 = torch.mul(argument_1, torch.expand_as(y, argument_1))
    input1 = torch.add(y1, argument_1, alpha=1)
    return (_28).forward(input1, )
class ConvTran(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  conv_deconv : __torch__.torch.nn.modules.conv.ConvTranspose2d
  bn : __torch__.torch.nn.modules.batchnorm.___torch_mangle_357.BatchNorm2d
  act : __torch__.torch.nn.modules.activation.___torch_mangle_358.SiLU
  def forward(self: __torch__.models.common.ConvTran,
    argument_1: Tensor) -> Tensor:
    _35 = self.act
    _36 = self.bn
    _37 = (self.conv_deconv).forward(argument_1, )
    _38 = (_35).forward((_36).forward(_37, ), )
    return _38
