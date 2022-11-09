import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Normal
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore import dtype as mstype
import pandas as pd
import numpy as np
import os
from numpy import *
import argparse

def load_data_V2(data_name):
    data_path = data_name
    data = pd.read_table('{path}'.format(path = data_path), sep=',', header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    id = data.pop(0)
    y = data.pop(1).astype('category')
    data_x = data.values
    
    data_y = np.zeros((data_x.shape[0],) ,dtype = np.int)
    idx = (y.values=='out')
    data_y[idx] = 1
    min_label = 1

    n_classes = int(max(data_y)+1)
    return data_x, data_y, min_label, n_classes

def load_data(data_name):
    data_path = os.path.join('./data/', data_name)
    data = pd.read_table('{path}'.format(path = data_path), sep=',', header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    id = data.pop(0)
    y = data.pop(1).astype('category')
    data_x = data.values
    data_y = y.cat.codes.values
    zeros_counts = (data_y==0).sum()
    ones_counts = (data_y==1).sum()
    min_label = 0 if zeros_counts<ones_counts else 1
    n_classes = int(max(data_y)+1)
    return data_x, data_y, min_label, n_classes


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
            tensor=initializer(Normal(self.var,self.mean), self.shape, mindspore.float32)
            #print("normal tensor",tensor)
        elif self.dist_type == 'categorical':
            uniform_int = ops.UniformInt()
            minval = Tensor(0, mstype.int32)
            maxval = Tensor(self.num_categories, mstype.int32)
            tensor = uniform_int(self.shape, minval, maxval)
            #tensor=np.random.randint(0,2,size=self.shape).reshape(self.shape[0],1)
            #tensor=Tensor(tensor.astype(np.int32))
            ops.stop_gradient(tensor)
            
        return tensor
    
    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        return new_obj


def prepare_z_y(G_batch_size, dim_z, nclasses, 
                fp16=False,z_var=1.0):
    z_ = Distribution(np.random.rand(G_batch_size, dim_z))
    z_.init_distribution('normal', mean=0, var=z_var)
    y_ = Distribution(np.zeros(G_batch_size))
    y_.init_distribution('categorical',num_categories=nclasses)
    return z_, y_





def active_sampling_V1(args, real_x, real_y, NetD_Ensemble, need_sample=True):
    if need_sample:
        pt = None
        for i in range(args.ensemble_num):
            netD = NetD_Ensemble[i]
            pt_i = netD(real_x, mode=2)   #get the confidence on real data
            if i==0:
                pt = pt_i
            else:
                pt += pt_i
        pt /= args.ensemble_num
        pt = pt.view(pt.shape[0],)
        batch_size_selected = int(real_x.shape[0]*args.active_rate)
        batch_size_selected = max(1, batch_size_selected)
        abs=ops.Abs()
        pt = abs(pt-0.5)  # select the instance with low margin value
        sort = ops.Sort(descending =False)
        _, idx = sort(pt)
        X = real_x[idx[0:batch_size_selected]]
        Y = real_y[idx[0:batch_size_selected]]
        X_unlabeled = real_x[idx[batch_size_selected:]]
        Y = Y.view(Y.shape[0],)
        X = X.view(X.shape[0], -1)
    else:
        batch_size_selected = int(real_x.shape[0]*args.active_rate)
        X = real_x[0:batch_size_selected]
        Y = real_y[0:batch_size_selected]
        X_unlabeled = None
    return X, Y, X_unlabeled


def loss_dis_real(dis_real, weights=None, gamma=2.0):
    softplus = ops.Softplus()
    exp = ops.Exp()
    cat1 = ops.Concat(1)
    mean = ops.ReduceMean(keep_dims=False)
    logpt = softplus(-dis_real)
    pt = exp(-logpt)
    if weights is None:
        p = pt*1
        p = p.view(len(dis_real), 1)
        p = (1-p)**gamma
        loss = p * logpt
        weights = pt
    else:
        weights = cat1((weights, pt))
        p = mean(weights, 1)
        p = p.view(len(dis_real), 1)
        p = (1-p)**gamma
        loss = p*logpt
    
    loss = mean(loss)
    return loss, weights

def loss_dis_fake(dis_fake, weights=None, gamma=2.0):
    softplus = ops.Softplus()
    exp = ops.Exp()
    cat1 = ops.Concat(1)
    mean = ops.ReduceMean(keep_dims=False)
    
    logpt = softplus(dis_fake)
    pt = exp(-logpt)

    if weights is None:
        p = pt*1
        p = p.view(len(dis_fake), 1)
        p = (1-p)**gamma
        loss = p * logpt
        weights = pt
    else:
        weights = cat1((weights, pt))
        p = mean(weights, 1)
        p = p.view(len(dis_fake), 1)
        p = (1-p)**gamma
        loss = p*logpt
    
    loss = mean(loss)
    
    return loss, weights



def join_strings(base_string, strings):
  return base_string.join([item for item in strings if item])

def save_weights(G, D_list, epoch, append_dict=None,weights_root='./weights'):

  root = weights_root
  if not os.path.exists(root):
    os.mkdir(root)
  print('Saving weights to %s...' % root)
  
  if append_dict!=None:
      mindspore.save_checkpoint(G, '%s/%s.ckpt' % (root, join_strings('_', ['G', str(epoch)])),append_dict=append_dict)
  else:
      mindspore.save_checkpoint(G, '%s/%s.ckpt' % (root, join_strings('_', ['G', str(epoch)])))
  for i,D in enumerate(D_list):
      mindspore.save_checkpoint(D, '%s/%s.ckpt' % (root, join_strings('_', [str(i),'D', str(epoch)])))

# Load a model's weights
def load_weights(G, D_list, weights_root="./weights",epoch=0,name_suffix=None, 
                 strict=False):
    
  root = weights_root
  print('Loading weights from %s...' % root)
  
  
  if G is not None:
    load_param_into_net(G,
      mindspore.load_checkpoint('%s/%s.ckpt' % (root, join_strings('_', ['G',str(epoch), name_suffix])),
      strict_load=strict))
  
  if D_list is not None:
    for i,D in enumerate(D_list):
        mindspore.load_checkpoint('%s/%s.ckpt' % (root, join_strings('_', [str(i),'D',str(epoch),name_suffix])),net=D,
        strict_load=strict)
  
  return G,D_list


def CSV_data_Loading(path):
    # loading data
    df = pd.read_csv(path) 
    
    labels = df['class']
    
    x_df = df.drop(['class'], axis=1)
    
    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)
    x = np.array(x)
    labels = np.array(labels)
    
    return x, labels;


def data_norm(dataset):
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals-minVals
    norm_data = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    norm_data = dataset - np.tile(minVals, (m,1))
    norm_data = norm_data / np.tile(ranges, (m, 1))
    return norm_data

