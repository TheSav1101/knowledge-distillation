from __future__ import absolute_import
from tensorflow.keras.applications import MobileNetV3Small
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import fractions
import os
import tensorflow as tf
import keras
import keras.layers as layers
from keras import ops
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#####INSERT VARIABLES HERE#####
base_folder = "./"
teacher_name = "models/SelecSLS60_statedict_better"
teacher_ext = ".pth"
joints_n = 17


######DISTILLER#########

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3, distillation_alpha=0.5):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.distillation_alpha = distillation_alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data

        teacher_predictions, teacher_heatmaps = self.teacher.forward_heats(x)
        with tf.GradientTape() as tape:
            student_predictions, student_heatmaps = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1)
            ) * (self.temperature ** 2)
            heatmaps_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_heatmaps / self.temperature, axis=1),
                tf.nn.softmax(student_heatmaps / self.temperature, axis=1)
            ) * (self.temperature ** 2)

            loss = self.alpha * student_loss + (1 - self.alpha) * (distillation_loss*(self.distillation_alpha) + heatmaps_loss*(1 - self.distillation_alpha))

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def test_step(self, data):
        x, y = data
        teacher_predictions = self.teacher(x, training=False)
        student_predictions = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, student_predictions)
        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
            tf.nn.softmax(student_predictions / self.temperature, axis=1)
        ) * (self.temperature ** 2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def call(self, inputs):
        return self.student(inputs)

##########################################################################
#########################TORCH MODEL######################################

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def bn_fuse(c, b):
    ''' BN fusion code adapted from my Caffe BN fusion code and code from @MIPT-Oulu. This function assumes everything is on the cpu'''
    with torch.no_grad():
        # BatchNorm params
        eps = b.eps
        mu = b.running_mean
        var = b.running_var
        gamma = b.weight

        if 'bias' in b.state_dict():
            beta = b.bias
        else:
            #beta = torch.zeros(gamma.size(0)).float().to(gamma.device)
            beta = torch.zeros(gamma.size(0)).float()

        # Conv params
        W = c.weight
        if 'bias' in c.state_dict():
            bias = c.bias
        else:
            bias = torch.zeros(W.size(0)).float()

        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

        W.mul_(A)
        bias.add_(b)

    return W.clone().detach(), bias.clone().detach()

class SelecSLSBlock(nn.Module):
    def __init__(self, inp, skip, k, oup, isFirst, stride):
        super(SelecSLSBlock, self).__init__()
        self.stride = stride
        self.isFirst = isFirst
        assert stride in [1, 2]

        #Process input with 4 conv blocks with the same number of input and output channels
        self.conv1 = nn.Sequential(
                nn.Conv2d(inp, k, 3, stride, 1,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k),
                nn.ReLU(inplace=True)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(k, k, 1, 1, 0,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k),
                nn.ReLU(inplace=True)
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(k, k//2, 3, 1, 1,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k//2),
                nn.ReLU(inplace=True)
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(k//2, k, 1, 1, 0,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k),
                nn.ReLU(inplace=True)
                )
        self.conv5 = nn.Sequential(
                nn.Conv2d(k, k//2, 3, 1, 1,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k//2),
                nn.ReLU(inplace=True)
                )
        self.conv6 = nn.Sequential(
                nn.Conv2d(2*k + (0 if isFirst else skip), oup, 1, 1, 0,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        assert isinstance(x,list)
        assert len(x) in [1,2]

        d1 = self.conv1(x[0])
        d2 = self.conv3(self.conv2(d1))
        d3 = self.conv5(self.conv4(d2))
        if self.isFirst:
            out = self.conv6(torch.cat([d1, d2, d3], 1))
            return [out, out]
        else:
            return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)) , x[1]]

class SelecSLSBlockFused(nn.Module):
    def __init__(self, inp, skip, a,b,c,d,e, oup, isFirst, stride):
        super(SelecSLSBlockFused, self).__init__()
        self.stride = stride
        self.isFirst = isFirst
        assert stride in [1, 2]

        #Process input with 4 conv blocks with the same number of input and output channels
        self.conv1 = nn.Sequential(
                nn.Conv2d(inp, a, 3, stride, 1,groups= 1, bias=True, dilation=1),
                nn.ReLU(inplace=True)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(a, b, 1, 1, 0,groups= 1, bias=True, dilation=1),
                nn.ReLU(inplace=True)
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(b, c, 3, 1, 1,groups= 1, bias=True, dilation=1),
                nn.ReLU(inplace=True)
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(c, d, 1, 1, 0,groups= 1, bias=True, dilation=1),
                nn.ReLU(inplace=True)
                )
        self.conv5 = nn.Sequential(
                nn.Conv2d(d, e, 3, 1, 1,groups= 1, bias=True, dilation=1),
                nn.ReLU(inplace=True)
                )
        self.conv6 = nn.Sequential(
                nn.Conv2d(a+c+e + (0 if isFirst else skip), oup, 1, 1, 0,groups= 1, bias=True, dilation=1),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        assert isinstance(x,list)
        assert len(x) in [1,2]

        d1 = self.conv1(x[0])
        d2 = self.conv3(self.conv2(d1))
        d3 = self.conv5(self.conv4(d2))
        if self.isFirst:
            out = self.conv6(torch.cat([d1, d2, d3], 1))
            return [out, out]
        else:
            return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)) , x[1]]

class Net(nn.Module):
    def __init__(self, nClasses=1000, config='SelecSLS60'):
        super(Net, self).__init__()

        #Stem
        self.stem = conv_bn(3, 32, 2)

        #Core Network
        self.features = []
        if config=='SelecSLS42':
            print('SelecSLS42')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 32,   0,  64,  64,  True,  2],
                [ 64,  64,  64, 128,  False, 1],
                [128,   0, 144, 144,  True,  2],
                [144, 144, 144, 288,  False, 1],
                [288,   0, 304, 304,  True,  2],
                [304, 304, 304, 480,  False, 1],
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(480, 960, 2),
                    conv_bn(960, 1024, 1),
                    conv_bn(1024, 1024, 2),
                    conv_1x1_bn(1024, 1280),
                    )
            self.num_features = 1280
        elif config=='SelecSLS42_B':
            print('SelecSLS42_B')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 32,   0,  64,  64,  True,  2],
                [ 64,  64,  64, 128,  False, 1],
                [128,   0, 144, 144,  True,  2],
                [144, 144, 144, 288,  False, 1],
                [288,   0, 304, 304,  True,  2],
                [304, 304, 304, 480,  False, 1],
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(480, 960, 2),
                    conv_bn(960, 1024, 1),
                    conv_bn(1024, 1280, 2),
                    conv_1x1_bn(1280, 1024),
                    )
            self.num_features = 1024
        elif config=='SelecSLS60':
            print('SelecSLS60')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 32,   0,  64,  64,  True,  2],
                [ 64,  64,  64, 128,  False, 1],
                [128,   0, 128, 128,  True,  2],
                [128, 128, 128, 128,  False, 1],
                [128, 128, 128, 288,  False, 1],
                [288,   0, 288, 288,  True,  2],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 416,  False, 1],
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(416, 756, 2),
                    conv_bn(756, 1024, 1),
                    conv_bn(1024, 1024, 2),
                    conv_1x1_bn(1024, 1280),
                    )
            self.num_features = 1280
        elif config=='SelecSLS60_B':
            print('SelecSLS60_B')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 32,   0,  64,  64,  True,  2],
                [ 64,  64,  64, 128,  False, 1],
                [128,   0, 128, 128,  True,  2],
                [128, 128, 128, 128,  False, 1],
                [128, 128, 128, 288,  False, 1],
                [288,   0, 288, 288,  True,  2],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 416,  False, 1],
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(416, 756, 2),
                    conv_bn(756, 1024, 1),
                    conv_bn(1024, 1280, 2),
                    conv_1x1_bn(1280, 1024),
                    )
            self.num_features = 1024
        elif config=='SelecSLS84':
            print('SelecSLS84')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 32,   0,  64,  64,  True,  2],
                [ 64,  64,  64, 144,  False, 1],
                [144,   0, 144, 144,  True,  2],
                [144, 144, 144, 144,  False, 1],
                [144, 144, 144, 144,  False, 1],
                [144, 144, 144, 144,  False, 1],
                [144, 144, 144, 304,  False, 1],
                [304,   0, 304, 304,  True,  2],
                [304, 304, 304, 304,  False, 1],
                [304, 304, 304, 304,  False, 1],
                [304, 304, 304, 304,  False, 1],
                [304, 304, 304, 304,  False, 1],
                [304, 304, 304, 512,  False, 1],
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(512, 960, 2),
                    conv_bn(960, 1024, 1),
                    conv_bn(1024, 1024, 2),
                    conv_1x1_bn(1024, 1280),
                    )
            self.num_features = 1280
        elif config=='SelecSLS102':
            print('SelecSLS102')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 32,   0,  64,  64,  True,  2],
                [ 64,  64,  64,  64,  False, 1],
                [ 64,  64,  64,  64,  False, 1],
                [ 64,  64,  64, 128,  False, 1],
                [128,   0, 128, 128,  True,  2],
                [128, 128, 128, 128,  False, 1],
                [128, 128, 128, 128,  False, 1],
                [128, 128, 128, 128,  False, 1],
                [128, 128, 128, 288,  False, 1],
                [288,   0, 288, 288,  True,  2],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 480,  False, 1],
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(480, 960, 2),
                    conv_bn(960, 1024, 1),
                    conv_bn(1024, 1024, 2),
                    conv_1x1_bn(1024, 1280),
                    )
            self.num_features = 1280
        else:
            raise ValueError('Invalid net configuration '+config+' !!!')

        #Build SelecSLS Core
        for inp, skip, k, oup, isFirst, stride  in self.selecSLS_config:
            self.features.append(SelecSLSBlock(inp, skip, k, oup, isFirst, stride))
        self.features = nn.Sequential(*self.features)

        #Classifier To Produce Inputs to Softmax
        self.classifier = nn.Sequential(
                nn.Linear(self.num_features, nClasses),
        )


    def forward(self, x):
        x = self.stem(x)
        x = self.features([x])
        x = self.head(x[0])
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        #x = F.log_softmax(x)
        return x

    def forward_heats(self, x):
        x = self.stem(x)
        x = self.features([x])
        x = self.head(x[0])
        y = x.clone()
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        #x = F.log_softmax(x)
        return x, y



    def prune_and_fuse(self, gamma_thresh, verbose=False):
        ''' Function that iterates over the modules in the model and prunes different parts by name. Sparsity emerges implicitly due to the use of
        adaptive gradient descent approaches such as Adam, in conjunction with L2 or WD regularization on the parameters. The filters
        that are implicitly zeroed out can be explicitly pruned without any impact on the model accuracy (and might even improve in some cases).
        '''
        #This function assumes a specific structure. If the structure of stem or head is changed, this code would need to be changed too
        #Also, this be ugly. Needs to be written better, but is at least functional
        #Perhaps one need not worry about the layers made redundant, they can be removed from storage by tracing with the JIT module??

        #We bring everything to the CPU, then later restore the device
        device = next(self.parameters()).device
        self.to("cpu")
        with torch.no_grad():
            #Assumes that stem is flat and has conv,bn,relu in order. Can handle one or more of these if one wants to deepen the stem.
            new_stem = []
            input_validity = torch.ones(3)
            for i in range(0,len(self.stem),3):
                input_size = sum(input_validity.int()).item()
                #Calculate the extent of sparsity
                out_validity  = abs(self.stem[i+1].weight) > gamma_thresh
                out_size = sum(out_validity.int()).item()
                W, b = bn_fuse(self.stem[i],self.stem[i+1])
                new_stem.append(nn.Conv2d(input_size, out_size, kernel_size = self.stem[i].kernel_size, stride=self.stem[i].stride, padding = self.stem[i].padding))
                new_stem.append(nn.ReLU(inplace=True))
                new_stem[-2].weight.copy_( torch.index_select(torch.index_select(W, 1, torch.nonzero(input_validity).squeeze()), 0, torch.nonzero(out_validity).squeeze()))
                new_stem[-2].bias.copy_(b[out_validity])
                input_validity = out_validity.clone().detach()
                if verbose:
                    print('Stem '+str(len(new_stem)/2 -1)+': Pruned '+str(len(out_validity) - out_size) + ' from '+str(len(out_validity)))
            self.stem = nn.Sequential(*new_stem)

            new_features = []
            skip_validity = 0
            for i in range(len(self.features)):
                inp = int(sum(input_validity.int()).item())
                if self.features[i].isFirst:
                    skip = 0
                a_validity = abs(self.features[i].conv1[1].weight) > gamma_thresh
                b_validity = abs(self.features[i].conv2[1].weight) > gamma_thresh
                c_validity = abs(self.features[i].conv3[1].weight) > gamma_thresh
                d_validity = abs(self.features[i].conv4[1].weight) > gamma_thresh
                e_validity = abs(self.features[i].conv5[1].weight) > gamma_thresh
                out_validity = abs(self.features[i].conv6[1].weight) > gamma_thresh

                new_features.append(SelecSLSBlockFused(inp, skip, int(sum(a_validity.int()).item()),int(sum(b_validity.int()).item()),int(sum(c_validity.int()).item()),int(sum(d_validity.int()).item()),int(sum(e_validity.int()).item()), int(sum(out_validity.int()).item()), self.features[i].isFirst, self.features[i].stride))

                #Conv1
                i_validity = input_validity.clone().detach()
                o_validity = a_validity.clone().detach()
                W, bias = bn_fuse(self.features[i].conv1[0], self.features[i].conv1[1])
                new_features[i].conv1[0].weight.copy_( torch.index_select(torch.index_select(W, 1, torch.nonzero(i_validity).squeeze()), 0, torch.nonzero(o_validity).squeeze()))
                new_features[i].conv1[0].bias.copy_(bias[o_validity])
                if verbose:
                    print('features.'+str(i)+'.conv1: Pruned '+str(len(o_validity) - sum(o_validity.int()).item()) + ' from '+str(len(o_validity)))
                #Conv2
                i_validity = o_validity.clone().detach()
                o_validity = b_validity.clone().detach()
                W, bias = bn_fuse(self.features[i].conv2[0], self.features[i].conv2[1])
                new_features[i].conv2[0].weight.copy_( torch.index_select(torch.index_select(W, 1, torch.nonzero(i_validity).squeeze()), 0, torch.nonzero(o_validity).squeeze()))
                new_features[i].conv2[0].bias.copy_(bias[o_validity])
                if verbose:
                    print('features.'+str(i)+'.conv2: Pruned '+str(len(o_validity) - sum(o_validity.int()).item()) + ' from '+str(len(o_validity)))
                #Conv3
                i_validity = o_validity.clone().detach()
                o_validity = c_validity.clone().detach()
                W, bias = bn_fuse(self.features[i].conv3[0], self.features[i].conv3[1])
                new_features[i].conv3[0].weight.copy_( torch.index_select(torch.index_select(W, 1, torch.nonzero(i_validity).squeeze()), 0, torch.nonzero(o_validity).squeeze()))
                new_features[i].conv3[0].bias.copy_(bias[o_validity])
                if verbose:
                    print('features.'+str(i)+'.conv3: Pruned '+str(len(o_validity) - sum(o_validity.int()).item()) + ' from '+str(len(o_validity)))
                #Conv4
                i_validity = o_validity.clone().detach()
                o_validity = d_validity.clone().detach()
                W, bias = bn_fuse(self.features[i].conv4[0], self.features[i].conv4[1])
                new_features[i].conv4[0].weight.copy_( torch.index_select(torch.index_select(W, 1, torch.nonzero(i_validity).squeeze()), 0, torch.nonzero(o_validity).squeeze()))
                new_features[i].conv4[0].bias.copy_(bias[o_validity])
                if verbose:
                    print('features.'+str(i)+'.conv4: Pruned '+str(len(o_validity) - sum(o_validity.int()).item()) + ' from '+str(len(o_validity)))
                #Conv5
                i_validity = o_validity.clone().detach()
                o_validity = e_validity.clone().detach()
                W, bias = bn_fuse(self.features[i].conv5[0], self.features[i].conv5[1])
                new_features[i].conv5[0].weight.copy_( torch.index_select(torch.index_select(W, 1, torch.nonzero(i_validity).squeeze()), 0, torch.nonzero(o_validity).squeeze()))
                new_features[i].conv5[0].bias.copy_(bias[o_validity])
                if verbose:
                    print('features.'+str(i)+'.conv5: Pruned '+str(len(o_validity) - sum(o_validity.int()).item()) + ' from '+str(len(o_validity)))
                #Conv6
                i_validity = torch.cat([a_validity.clone().detach(), c_validity.clone().detach(), e_validity.clone().detach()], 0)
                if self.features[i].isFirst:
                    skip = int(sum(out_validity.int()).item())
                    skip_validity = out_validity.clone().detach()
                else:
                    i_validity = torch.cat([i_validity, skip_validity], 0)
                o_validity = out_validity.clone().detach()
                W, bias = bn_fuse(self.features[i].conv6[0], self.features[i].conv6[1])
                new_features[i].conv6[0].weight.copy_( torch.index_select(torch.index_select(W, 1, torch.nonzero(i_validity).squeeze()), 0, torch.nonzero(o_validity).squeeze()))
                new_features[i].conv6[0].bias.copy_(bias[o_validity])
                if verbose:
                    print('features.'+str(i)+'.conv6: Pruned '+str(len(o_validity) - sum(o_validity.int()).item()) + ' from '+str(len(o_validity)))

                input_validity = out_validity.clone().detach()
            self.features = nn.Sequential(*new_features)

            new_head = []
            for i in range(len(self.head)):
                input_size = int(sum(input_validity.int()).item())
                #Calculate the extent of sparsity
                out_validity  = abs(self.head[i][1].weight) > gamma_thresh
                out_size = int(sum(out_validity.int()).item())
                W, b = bn_fuse(self.head[i][0],self.head[i][1])
                new_head.append(nn.Conv2d(input_size, out_size, kernel_size = self.head[i][0].kernel_size, stride=self.head[i][0].stride, padding = self.head[i][0].padding))
                new_head.append(nn.ReLU(inplace=True))
                new_head[-2].weight.copy_( torch.index_select(torch.index_select(W, 1, torch.nonzero(input_validity).squeeze()), 0, torch.nonzero(out_validity).squeeze()))
                new_head[-2].bias.copy_(b[out_validity])
                input_validity = out_validity.clone().detach()
                if verbose:
                    print('Head '+str(len(new_head)/2 -1)+': Pruned '+str(len(out_validity) - out_size) + ' from '+str(len(out_validity)))
            self.head = nn.Sequential(*new_head)

            new_classifier = []
            new_classifier.append(nn.Linear(int(sum(input_validity.int()).item()), self.classifier[0].weight.shape[0]))
            new_classifier[0].weight.copy_(torch.index_select(self.classifier[0].weight, 1, torch.nonzero(input_validity).squeeze()))
            new_classifier[0].bias.copy_(self.classifier[0].bias)
            self.classifier = nn.Sequential(*new_classifier)

        self.to(device)
        
        
###############################END TORCH MODEL#########################################
#######################################################################################


#######SETUP TEACHER########
teacher = Net()
teacher.load_state_dict(torch.load(colab_folder + teacher_name + teacher_ext))


mobilenet_base = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
mobilenet_layers = mobilenet_base.layers
selected_layers = mobilenet_layers #[:115] #Cut as in paper best = 86
student_backbone = keras.Model(mobilenet_base.input, selected_layers[-1].output, name="student_backbone")
for layer in student_backbone.layers:
    layer.trainable = False


#block 13_a
top = layers.Conv2D(368, 1, activation="relu")(student_backbone.output)
top = layers.DepthwiseConv2D(3, activation="relu", padding="same")(top)
top = layers.Conv2D(256, 1)(top) #### NO RELU
bot = layers.Conv2D(256, 1)(student_backbone.output)
block_13_a_output = layers.add([top, bot])

#block 13_b
top = layers.Conv2D(192, 1, activation="relu")(block_13_a_output)
top = layers.DepthwiseConv2D( 3, activation="relu", padding="same")(top)
block_13_b_output = layers.Conv2D(192, 1, activation="relu")(top)

top = layers.UpSampling2D(interpolation="bilinear")(block_13_b_output)
top = layers.Conv2D(128, 3, activation="relu")(top)

mid = layers.UpSampling2D(interpolation="bilinear")(block_13_b_output)
mid = layers.Conv2D(3 * joints_n, 3)(mid)

class JointLength(layers.Layer):
    def call(self, x):
        delta_x, delta_y, delta_z = tf.split(x, num_or_size_splits=3, axis=3)
        return tf.abs(delta_x) + tf.abs(delta_y) + tf.abs(delta_z)

bl = JointLength()(mid)

concat = layers.Concatenate()([top, mid, bl])
fin = layers.Conv2D(128, 1, activation="relu")(concat)
fin = layers.DepthwiseConv2D(3, activation="relu", padding="same")(fin)
fin = layers.Conv2D(4 * joints_n, 1)(fin)

class HeatmapLayer(layers.Layer):
    def call(self, x):
      H,X,Y,Z = tf.split(x,num_or_size_splits=4, axis=3)
      return [H,X,Y,Z]

class OutputLayer(layers.Layer):
    def call(self, x):
      outs = []
      for y in x:
        out = layers.Flatten()(y)
        out = layers.Dense(joints_n)(out)
        outs.append(out)
      return outs

st_heats = HeatmapLayer()(fin)
st_output = OutputLayer()(st_heats)
student = keras.Model(student_backbone.input, [st_output, st_heats], name = "student")
student.summary()
student_scratch = keras.models.clone_model(student)


import matplotlib.pyplot as plt
def plot_history(history):
  print(history.history.keys())
  #  "Accuracy"
  plt.plot(history.history['sparse_categorical_accuracy'])
  plt.plot(history.history['val_sparse_categorical_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()
  # "Loss"
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()
  
  
##################################################
#################DATA SETUP#######################



with open('h36m_train.pkl', 'rb') as f:
    dataset_train = pickle.load(f)
    
images = []
labels = []

i=0
for bb in dataset_train:
    img = cv2.imread('./images/' + bb['image'])
    box = bb['box']
    boxed_img = img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
    x = cv2.resize(boxed_img, (224, 224)) 
    y = bb['joints_3d']
    
    images.append(x)
    labels.append(y)
    i+=1
    if(i >=100):
        break

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


##################################################
#####################TRAINING#####################
  
# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.5,
    temperature=3,
)

# Distill teacher to student
history = distiller.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), shuffle=True)
plot_history(history)


# Train student as doen usually
student_scratch.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate student trained from scratch.
history = student_scratch.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), shuffle=True)
plot_history(history)