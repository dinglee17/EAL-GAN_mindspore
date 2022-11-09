import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Normal
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net

import pandas as pd
import numpy as np
import os
from numpy import *

class Distribution(mindspore.Tensor):
  # Init the params of the distribution
  def init_distribution(self, dist_type, **kwargs):    
    self.dist_type = dist_type
    self.dist_kwargs = kwargs
    if self.dist_type == 'normal':
      self.mean, self.var = kwargs['mean'], kwargs['var']


    elif self.dist_type == 'categorical':
      self.num_categories = kwargs['num_categories']

  def sample_(self):
    if self.dist_type == 'normal':
      print("get self.mean", self.mean)
      print("get self.var", self.var)
      #self.normal_(self.mean, self.var)
      #normal = mindspore.ops.normal(mean=self.mean, stddev=self.var)
      #self.ops()

    elif self.dist_type == 'categorical':
      print("get self.num_categories", self.num_categories)
      #self.random_(0, self.num_categories)
      variable = random.randint(0, self.num_categories)
      return Tensor(variable)
    # return self.variable
    
  # Silly hack: overwrite the to() method to wrap the new object
  # in a distribution as well
  def to(self, *args, **kwargs):
    new_obj = Distribution(self)
    new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
    new_obj.data = super().to(*args, **kwargs)    
    return new_obj


# Convenience function to prepare a z and y vector
def prepare_z_y(G_batch_size, dim_z, nclasses,fp16=False,z_var=1.0):
  standnormal = mindspore.ops.StandardNormal(seed=2)
  out = standnormal((G_batch_size, dim_z))
  z_ = Distribution(mindspore.Tensor(out))
  z_.init_distribution('normal', mean=0, var=z_var)
  if fp16:
    z_ = z_.half()

  zeros = mindspore.ops.Zeros()
  y_ = zeros(G_batch_size, mindspore.float32)
  y_ = Distribution(y_)
  y_.init_distribution('categorical',num_categories=nclasses)
  return z_, y_