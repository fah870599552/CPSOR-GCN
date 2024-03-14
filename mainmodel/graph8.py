from __future__ import print_function
import tensorflow as tf
from keras import activations, initializers, constraints
from keras import regularizers
#from keras.engine import Layer
from tensorflow.keras.layers import Layer,concatenate
import keras.backend as K
import tracemalloc
# from recompute import *
import sys
from distutils.util import strtobool
import scipy.sparse as sp
import time

class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, #[N,F]+[N,F]新[（？， 256， 1），（?,256,256），（?,256,256），（?,256,256）]
                 units,  #16
                 support=3,#3

                 activation=None,#'RELU'
                 use_bias=None,
                 kernel_initializer='glorot_uniform', #Gaussian distribution L2(5e-4)
                 bias_initializer='zeros',
                 kernel_regularizer=None, 
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # 施加在权重上的正则项
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # 施加在偏置向量上的正则项
        self.bias_regularizer = regularizers.get(bias_regularizer)
        # 施加在输出上的正则项
        self.activity_regularizer = regularizers.get(activity_regularizer)
        # 对主权重矩阵进行约束
        self.kernel_constraint = constraints.get(kernel_constraint)
         # 对偏置向量进行约束
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True
        # self.output_dim = output_dim
        self.support = support
        assert support >= 1

    def compute_output_shape(self, input_shapes):#如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断
        # 特征矩阵形状#[N,F]新[?,255,1]
        features_shape = input_shapes[0]#[None,F]新[?, N, F][?,255,1]
        # 输出形状为(批大小, 输出维度)[B, 16]新[Batch, 256, 16]
        output_shape = (features_shape[0], 2, self.units)
        return output_shape  # (batch_size, output_dim)[B, 16]新[Batch, 256, 16]
     #input shape[(None, 1433),(None,None),(None,None),(None, None)]
     #新input shape[(None, 256，1),(None,None),(None,None),(None, None)]
    def build(self, input_shapes):#这是定义权重的方法，可训练的权应该在这里被加入列
        # print('input_shapes',input_shapes)
        features_shape = input_shapes[0]#[None,1433]新[None, 256，1]
        # assert len(features_shape) == 3
        input_dim = features_shape[3]#1433新1
        # print('features_shape1',features_shape)
        # print('input_dim',input_dim)
        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),#(1433*3, 16)新(1*3, 16)
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(256, self.units,),#(？，16)新（？，256，16）
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    # @recompute_grad
    def call(self, inputs, mask=None):#这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心call的第一个参数：输入张量
        #现将[(?,256,256),(?,256,256),(?,256,256)]转变为[(256, 256), (256, 256), (256, 256)]
        features = inputs[0]#[N,F][?, F]新[?,256,1]
        seq_len = features.shape[1]
        results = []
        dim3 = 0
        dim4 = 0
        for k in range(seq_len):
            basis = list()
            binput = inputs[1][0]
            for s in range(3):
                B1 = binput[s][:,k,:,:][0]
                basis.append(B1)
            features_iter = features[:,k,:,:]
            supports = list()#(?,4399)
            for i in range(self.support):
                #A*X
                #old: [2708,2708]*[2708,1433]=[2708, 1433][?,1433]
                #NEW: [256,256]*[?,256,1]=[?,256,1]
                AAA = basis[i]
                AX = tf.einsum("ij,ljk->lik", AAA, features_iter)
                supports.append(AX)
            supports = K.concatenate(supports, axis=2)
            #old:[?,4299]
            #new:[?, 256, 3]
            #A*X*W
            output = tf.einsum("lij,jk->lik", supports, self.kernel)
            #old:(?, 16)，supports=[?,4399][2708,4399],self.kernel[4399, 16]
            #new:(？，256，16),supports=[?,256,3],self.kernel[3,16],要得到的是[?,256,16]
            if self.bias:
                #A*X*W+b
                output += self.bias
            myout = self.activation(output)
            dim3 = myout.shape[1]
            dim4 = myout.shape[2]
            results.append(myout)
        result = concatenate(results, axis=1)

        result = tf.expand_dims(result, axis=1)
        result = K.reshape(result,(-1,seq_len,dim3,dim4))
        # tracemalloc.stop()
        # memory_used, peak_memory = tracemalloc.get_traced_memory()
        # print(f"Memory used: {memory_used / 1024:.2f}KB")
        # print(f"Peak memory usage: {peak_memory / 1024:.2f}KB")
        return result#非线性激活

    def get_config(self):

        config = super(GraphConvolution, self).get_config()
        config['units'] = self.units
        return config

# 定义一个继承自 GraphConvolution 的名为 TwoLayerGraphConvolution 的新类：

class TwoLayerGraphConvolution(GraphConvolution):
    def __init__(self, units, support=3, activation=None, use_bias=None,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        super(TwoLayerGraphConvolution, self).__init__(units, support, activation, use_bias,
                                                       kernel_initializer, bias_initializer,
                                                       kernel_regularizer, bias_regularizer,
                                                       activity_regularizer, kernel_constraint,
                                                       bias_constraint, **kwargs)

# 覆盖构建方法以定义第二层的权重和偏差：
    def build(self, input_shapes):
        super(TwoLayerGraphConvolution, self).build(input_shapes)
        input_dim = self.units

        self.kernel_2 = self.add_weight(shape=(input_dim, self.units),  # (16, 16)
                                        initializer=self.kernel_initializer,
                                        name='kernel_2',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias_2 = self.add_weight(shape=(self.units,),  # (16,)
                                          initializer=self.bias_initializer,
                                          name='bias_2',
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint)
        else:
            self.bias_2 = None

# 覆盖调用方法以包括第二层的计算：
    def call(self, inputs, mask=None):
        output_1 = super(TwoLayerGraphConvolution, self).call(inputs, mask)
        output_2 = K.dot(output_1, self.kernel_2)
        if self.bias_2:
            output_2 += self.bias_2
        return self.activation(output_2)

# 更新 compute_output_shape 方法以反映第二层的输出形状：
    def compute_output_shape(self, input_shapes):
        output_shape_1 = super(TwoLayerGraphConvolution, self).compute_output_shape(input_shapes)
        output_shape_2 = (output_shape_1[0], output_shape_1[1], self.units)  # (batch_size, 2, 16)
        return output_shape_2

# 最后 更新 get_config 方法以包含新层的配置：
    def get_config(self):
        config = super(TwoLayerGraphConvolution, self).get_config()
        config['units'] = self.units
        return config
