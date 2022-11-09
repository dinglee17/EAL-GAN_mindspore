import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Normal
import mindspore.ops as ops
from mindspore import Tensor
import pandas as pd
import numpy as np
import os


def loss_dis_real(dis_real, out_category, y, weights=None, cat_weight=None, gamma=2.0):
    #step 1: the loss for GAN
    softplus = ops.Softplus()
    exp = ops.Exp()
    cat1 = ops.Concat(1)
    mean = ops.ReduceMean(keep_dims=False)
    cast = ops.Cast()
    log = ops.Log()
    
    logpt = softplus(-dis_real)
    pt = exp(-logpt)
    if weights is None:
        weights = pt
        p = pt*1
        p = p.view(dis_real.shape[0], 1)
        p = (1-p)**gamma
        loss = p * logpt
    else:
        weights = cat1((weights, pt))
        p = mean(weights, 1)
        p = p.view(len(dis_real), 1)
        p = (1-p)**gamma
        loss = p*logpt
    loss = mean(loss)

    #step 2: loss for classifying
    target = y.view(y.shape[0], 1)
    target = target
    out_category = out_category
    pt_cat = (1.-cast(target,mindspore.float32))*(1-out_category) + cast(target,mindspore.float32)*out_category
    logpt_cat = -log(pt_cat)
    batch_size = target.shape[0]
    
    if cat_weight is None:
        cat_weight = pt_cat
        p = pt_cat*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    else:
        cat_weight = cat1((cat_weight, pt_cat))
        p = mean(cat_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    logpt_cat = p*logpt_cat
    loss_cat = mean(logpt_cat)
    return loss, loss_cat, weights, cat_weight

def loss_dis_real_ave_weights(dis_real, out_category, y, dis_index=0,weights=None, cat_weight=None, gamma=2.0):
    #step 1: the loss for GAN
    softplus = ops.Softplus()
    exp = ops.Exp()
    cat1 = ops.Concat(1)
    mean = ops.ReduceMean(keep_dims=False)
    cast = ops.Cast()
    log = ops.Log()
    
    logpt = softplus(-dis_real)
    pt = exp(-logpt)
    if weights is None:
        weights = pt
        p = pt*1
        p = p.view(dis_real.shape[0], 1)
        p = (1-p)**gamma
        loss = p * logpt
    else:
        weights = (weights*(dis_index)+pt)/(dis_index+1)
        #p = mean(weights, 1)
        p=weights
        p = p.view(len(dis_real), 1)
        p = (1-p)**gamma
        loss = p*logpt
    loss = mean(loss)

    #step 2: loss for classifying
    target = y.view(y.shape[0], 1)
    target = target
    out_category = out_category
    pt_cat = (1.-cast(target,mindspore.float32))*(1-out_category) + cast(target,mindspore.float32)*out_category
    logpt_cat = -log(pt_cat)
    batch_size = target.shape[0]
    
    if cat_weight is None:
        cat_weight = pt_cat
        p = pt_cat*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    else:
        cat_weight = (cat_weight*(dis_index)+pt_cat)/(dis_index+1)
        #p = mean(cat_weight, 1)
        p = cat_weight
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    logpt_cat = p*logpt_cat
    loss_cat = mean(logpt_cat)
    return loss, loss_cat, weights, cat_weight


def loss_dis_fake(dis_fake, out_category, y, weights=None, cat_weight=None, gamma=2.0):
    #step 1: the loss for GAN
    softplus = ops.Softplus()
    exp = ops.Exp()
    cat1 = ops.Concat(1)
    mean = ops.ReduceMean(keep_dims=False)
    cast = ops.Cast()
    log = ops.Log()
    
    logpt = softplus(dis_fake)
    pt = exp(-logpt)

    if weights is None:
        p = pt*1
        p = p.view(len(dis_fake), 1)
        p = (1-p)**gamma
        loss = p* logpt
        weights = pt
    else:
        weights = cat1((weights, pt))
        p = mean(weights, 1)
        p = p.view(len(dis_fake), 1)
        p = (1-p)**gamma
        loss = p*logpt
    
    loss = mean(loss)

    #step 2: loss for classifying
    target = y.view(y.shape[0], 1)
    pt_cat = (1.-cast(target,mindspore.float32))*(1-out_category) + cast(target,mindspore.float32)*out_category
    logpt_cat = -log(pt_cat)
    batch_size = target.shape[0]
    
    if cat_weight is None:
        cat_weight = pt_cat
        p = pt_cat*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    else:
        cat_weight = cat1((cat_weight, pt_cat))
        p = mean(cat_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    logpt_cat = p*logpt_cat
    loss_cat = mean(logpt_cat)
    return loss, loss_cat, weights, cat_weight


def loss_dis_fake_ave_weights(dis_fake, out_category, y, dis_index=0, weights=None, cat_weight=None, gamma=2.0):
    #step 1: the loss for GAN
    softplus = ops.Softplus()
    exp = ops.Exp()
    cat1 = ops.Concat(1)
    mean = ops.ReduceMean(keep_dims=False)
    cast = ops.Cast()
    log = ops.Log()
    
    logpt = softplus(dis_fake)
    pt = exp(-logpt)

    if weights is None:
        p = pt*1
        p = p.view(len(dis_fake), 1)
        p = (1-p)**gamma
        loss = p* logpt
        weights = pt
    else:
        weights = (weights*(dis_index)+pt)/(dis_index+1)
        p = weights
        #p = mean(weights, 1)
        p = p.view(len(dis_fake), 1)
        p = (1-p)**gamma
        loss = p*logpt
    
    loss = mean(loss)

    #step 2: loss for classifying
    target = y.view(y.shape[0], 1)
    pt_cat = (1.-cast(target,mindspore.float32))*(1-out_category) + cast(target,mindspore.float32)*out_category
    logpt_cat = -log(pt_cat)
    batch_size = target.shape[0]
    
    if cat_weight is None:
        cat_weight = pt_cat
        p = pt_cat*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    else:
        cat_weight = (cat_weight*(dis_index)+pt_cat)/(dis_index+1)
        #p = mean(cat_weight, 1)
        p = cat_weight
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    logpt_cat = p*logpt_cat
    loss_cat = mean(logpt_cat)
    return loss, loss_cat, weights, cat_weight