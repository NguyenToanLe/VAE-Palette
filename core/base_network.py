import torch.nn as nn
from .types_ import *
from abc import abstractmethod


class BaseNetwork(nn.Module):
  def __init__(self, init_type='kaiming', gain=0.02):
    super(BaseNetwork, self).__init__()
    self.init_type = init_type
    self.gain = gain

  def init_weights(self):
    """
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    """
    
    def init_func(m):
      classname = m.__class__.__name__
      if classname.find('InstanceNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
          nn.init.constant_(m.weight.data, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)
      elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if self.init_type == 'normal':
          nn.init.normal_(m.weight.data, 0.0, self.gain)
        elif self.init_type == 'xavier':
          nn.init.xavier_normal_(m.weight.data, gain=self.gain)
        elif self.init_type == 'xavier_uniform':
          nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif self.init_type == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif self.init_type == 'orthogonal':
          nn.init.orthogonal_(m.weight.data, gain=self.gain)
        elif self.init_type == 'none':  # uses pytorch's default init method
          m.reset_parameters()
        else:
          raise NotImplementedError('initialization method [%s] is not implemented' % self.init_type)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)

    self.apply(init_func)
    # propagate to children
    for m in self.children():
      if hasattr(m, 'init_weights'):
        m.init_weights(self.init_type, self.gain)


class BaseVAE(nn.Module):

    def __init__(self, init_type='kaiming', gain=0.02) -> None:
      super(BaseVAE, self).__init__()
      self.init_type = init_type
      self.gain = gain

    def encode(self, input: Tensor) -> List[Tensor]:
      raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
      raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
      raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
      raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
      pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
      pass

    def init_weights(self):
      """
      initialize network's weights
      init_type: normal | xavier | kaiming | orthogonal
      https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
      """

      def init_func(m):
        classname = m.__class__.__name__
        if classname.find('InstanceNorm2d') != -1:
          if hasattr(m, 'weight') and m.weight is not None:
            nn.init.constant_(m.weight.data, 1.0)
          if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
          if self.init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, self.gain)
          elif self.init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=self.gain)
          elif self.init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight.data, gain=1.0)
          elif self.init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
          elif self.init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=self.gain)
          elif self.init_type == 'none':  # uses pytorch's default init method
            m.reset_parameters()
          else:
            raise NotImplementedError('initialization method [%s] is not implemented' % self.init_type)
          if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

      self.apply(init_func)
      # propagate to children
      for m in self.children():
        if hasattr(m, 'init_weights'):
          m.init_weights(self.init_type, self.gain)

    