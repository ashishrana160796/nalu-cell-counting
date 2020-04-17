# import statements for model design and training.
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import (
    Input,
    Activation,
    concatenate,
    add,
    merge,
    Dropout,
    Reshape,
    Permute,
    Dense,
    UpSampling2D,
    Flatten
    )
from keras.optimizers import SGD
from keras.layers.convolutional import (
    Convolution2D)
from keras.layers.pooling import (
    MaxPooling2D,
    AveragePooling2D
    )
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import numpy as np
import keras.backend as K
from keras.layers import *
from keras.initializers import *
from keras.models import *
from keras import initializers

# Code is referenced from: [kgrm](https://github.com/kgrm)
class NALU(Layer):
    def __init__(self, units, MW_initializer='glorot_uniform',
                 G_initializer='glorot_uniform', mode="NALU",
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NALU, self).__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.MW_initializer = initializers.get(MW_initializer)
        self.G_initializer = initializers.get(G_initializer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.W_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.MW_initializer,
                                     name='W_hat')
        self.M_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.MW_initializer,
                                     name='M_hat')
        if self.mode == "NALU":
            self.G = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.G_initializer,
                                     name='G')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        W = K.tanh(self.W_hat) * K.sigmoid(self.M_hat)
        a = K.dot(inputs, W)
        if self.mode == "NAC":
            output = a
        elif self.mode == "NALU":
            m = K.exp(K.dot(K.log(K.abs(inputs) + 1e-7), W))
            g = K.sigmoid(K.dot(K.abs(inputs), self.G))
            output = g * a + (1 - g) * m
        else:
            raise ValueError("Valid modes: 'NAC', 'NALU'.")
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

# weight decay defined on model's parametric matrices for regularization purpose.
weight_decay = 1e-5

# Models that can be trained using this script.
# 1. Regular FCRN model.
# 2. FCRN model with NALU/NAC Unit.
# 3. Regular U-net Model.
# 4. U-net model with NAC/NALU Unit.

def _conv_bn_relu(nb_filter, row, col, subsample = (1,1)):
    def f(input):
        conv_a = Convolution2D(nb_filter, row, col, subsample = subsample,
                               init = 'orthogonal', 
                               border_mode='same', bias = False)(input)
        norm_a = BatchNormalization()(conv_a)
        # 1. This actionvation function can be changed as demonstrated in experiments to create FCRN/U-net variants.
        act_a = Activation(activation = 'relu')(norm_a)
        return act_a
    return f

def _conv_bn_lin(nb_filter, row, col, subsample = (1,1)):
    def f(input):
        conv_a = Convolution2D(nb_filter, row, col, subsample = subsample,
                               init = 'orthogonal', 
                               border_mode='same', bias = False)(input)
        norm_a = BatchNormalization()(conv_a)
        # 2. For demonstration a linear batch normalization layer created that can also be used.
        act_a = Activation(activation = 'linear')(norm_a)
        return act_a
    return f
    
        
def _conv_bn_relu_x2(nb_filter, row, col, subsample = (1,1)):
    def f(input):
        # 1. batch normalization with relu activation, again it can be changed to create different variants of FCRN.
        conv_a = Convolution2D(nb_filter, row, col, subsample = subsample,
                               init = 'orthogonal', border_mode = 'same',bias = False,
                               W_regularizer = l2(weight_decay),
                               b_regularizer = l2(weight_decay))(input)
        norm_a = BatchNormalization()(conv_a)
        act_a = Activation(activation = 'relu')(norm_a)
        conv_b = Convolution2D(nb_filter, row, col, subsample = subsample,
                              init = 'orthogonal', border_mode = 'same',bias = False,
                              W_regularizer = l2(weight_decay),
                              b_regularizer = l2(weight_decay))(act_a)
        norm_b = BatchNormalization()(conv_b)
        act_b = Activation(activation = 'relu')(norm_b)
        return act_a
    return f
    

# 1. FCRN base and FCRN/NALU base model structure decleration.


def fcrn_base(input):
    # This model contains convolutional operation with 2x batch normalization function.
    # One time batch normalization can also opted, if required. Change _conv_bn_relu_x2 to _conv_bn_relu
    block1 = _conv_bn_relu_x2(32,3,3)(input)
    pool1 = MaxPooling2D(pool_size=(2,2))(block1)
    # =========================================================================
    block2 = _conv_bn_relu_x2(64,3,3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
    # =========================================================================
    block3 = _conv_bn_relu_x2(128,3,3)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
    # =========================================================================
    block4 = _conv_bn_relu(512,3,3)(pool3)
    # =========================================================================
    up5 = UpSampling2D(size=(2, 2))(block4)
    block5 = _conv_bn_relu_x2(128,3,3)(up5)
    # =========================================================================
    up6 = UpSampling2D(size=(2, 2))(block5)
    block6 = _conv_bn_relu_x2(64,3,3)(up6)
    # =========================================================================
    up7 = UpSampling2D(size=(2, 2))(block6)
    block7 = _conv_bn_relu_x2(32,3,3)(up7)
    return block7

def fcrn_nalu(input):

    # input-(256,256,3) or input_, output-(128,128,32)
    block1 = _conv_bn_relu_x2(32,3,3)(input)
    pool1 = MaxPooling2D(pool_size=(2,2))(block1)   
    nal1 = NALU(32, mode="NAC", 
             MW_initializer=RandomNormal(stddev=1),
             G_initializer=Constant(10))(block1) # volume- (256,256,32)
    # =========================================================================
    # input-(128,128,32), output-(64,64,64)
    block2 = _conv_bn_relu_x2(64,3,3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
    nal2 = NALU(64, mode="NAC", 
             MW_initializer=RandomNormal(stddev=1),
             G_initializer=Constant(10))(block2) # volume- (128,128,64)
    # ========================================================================= 
    # input-(64,64,64), output-(32,32,128)
    block3 = _conv_bn_relu_x2(128,3,3)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
    nal3 = NALU(128, mode="NAC", 
             MW_initializer=RandomNormal(stddev=1),
             G_initializer=Constant(10))(block3) # volume- (64,64,128)
    # ========================================================================= 
    # input-(32,32,512), output-(64,64,128)# input-(32,32,128), output-(32,32,512)
    block4 = _conv_bn_relu(512,3,3)(pool3)
    # =========================================================================
    # input-(32,32,512), output-(64,64,512)
    up5 =  UpSampling2D(size=(2, 2))(block4)
    # inputs-(64,64,512), (64,64,128)  output-(64,64,640)
    block5 = concatenate([_conv_bn_relu_x2(128,3,3)(up5), nal3])
    # inputs-(64,64,640), ->(64,64,128)  output-(64,64,128)
    block5 = _conv_bn_relu_x2(128,3,3)(up5)
    # ========================================================================= 
    # input-(64,64,128), output-(128,128,128)
    up6 = UpSampling2D(size=(2, 2))(block5)
    # inputs-(128,128,128), ->(128,128,64)  output-(128,128,192)
    block6 = concatenate([_conv_bn_relu_x2(64,3,3)(up6), nal2])
    # input-(128,128,192), output-(128,128,64)
    block6 = _conv_bn_relu_x2(64,3,3)(up6)
    # =========================================================================
    # Compressing the dimension analysis going forward with upcoming layers & for U-net also.
    # input-(64,64,128), output-(128,128,128)
    up7 = UpSampling2D(size=(2, 2))(block6)
    block7 = concatenate([_conv_bn_relu_x2(32,3,3)(up7), nal1])
    block7 = _conv_bn_relu_x2(32,3,3)(up7)

    return block7
    
    
# 1. U-net base and U-net/NALU base model structure decleration.

def U_net_base(input, nb_filter = 64):
    block1 = _conv_bn_relu_x2(nb_filter,3,3)(input)
    pool1 = MaxPooling2D(pool_size=(2,2))(block1)
    # =========================================================================
    block2 = _conv_bn_relu_x2(128,3,3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
    # =========================================================================
    block3 = _conv_bn_relu_x2(256,3,3)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
    # =========================================================================
    block4 = _conv_bn_relu_x2(256,3,3)(pool3)
    up4 = merge([UpSampling2D(size=(2, 2))(block4), block3], mode='concat', concat_axis=-1)
    # =========================================================================
    block5 = _conv_bn_relu_x2(128,3,3)(up4)
    up5 = merge([UpSampling2D(size=(2, 2))(block5), block2], mode='concat', concat_axis=-1)
    # =========================================================================
    block6 = _conv_bn_relu_x2(nb_filter,3,3)(up5)
    up6 = merge([UpSampling2D(size=(2, 2))(block6), block1], mode='concat', concat_axis=-1)
    # =========================================================================
    block7 = _conv_bn_relu(32,3,3)(up6)
    return block7

def u_net_nalu(input, nb_filter = 64):
    # input-(256,256,3), output-(128,128,64)
    block1 = _conv_bn_relu_x2(64,3,3)(input)
    pool1 = MaxPooling2D(pool_size=(2,2))(block1)
    nal1 = NALU(64, mode="NAC", 
             MW_initializer=RandomNormal(stddev=1),
             G_initializer=Constant(10))(block1) # volume- (256,256,64)
    # =========================================================================
    # input-(128,128,64), output-(64,64,128)
    block2 = _conv_bn_relu_x2(128,3,3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
    nal2 = NALU(128, mode="NAC", 
             MW_initializer=RandomNormal(stddev=1),
             G_initializer=Constant(10))(block2) # volume- (128,128,128)
    # ========================================================================= 
    # input-(64,64,128), output-(32,32,256)
    block3 = _conv_bn_relu_x2(256,3,3)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
    nal3 = NALU(256, mode="NAC", 
             MW_initializer=RandomNormal(stddev=1),
             G_initializer=Constant(10))(block3) # volume- (64,64,256)
    # =========================================================================
    # input-(32,32,256), output-(64,64,256)
    block4 = _conv_bn_relu_x2(256,3,3)(pool3)
    up4 = concatenate([UpSampling2D(size=(2, 2))(block4), block3], axis=-1)
    up4 = concatenate([up4, nal3], axis=-1)
    up4 = _conv_bn_relu_x2(256,3,3)(up4)
    # =========================================================================
    # input-(64,64,256), output-(128,128,128)
    block5 = _conv_bn_relu_x2(128,3,3)(up4)
    up5 = concatenate([UpSampling2D(size=(2, 2))(block5), block2], axis=-1)
    up5 = concatenate([up5, nal2], axis=-1)
    up5 = _conv_bn_relu_x2(128,3,3)(up5)
    # =========================================================================
    # input-(128,128,128), output-(256,256,64)
    block6 = _conv_bn_relu_x2(64,3,3)(up5)
    up6 = concatenate([UpSampling2D(size=(2, 2))(block6), block1], axis=-1) # input-128, output-256
    up6 = concatenate([up6, nal1], axis=-1)
    up6 = _conv_bn_relu_x2(64,3,3)(up6)
    # =========================================================================
    # input-(256,256,64), output-(256,256,32)
    block7 = _conv_bn_relu(32,3,3)(up6)
    return block7
    
def buildmodel_fcrn_nalu (input_dim):
    input_ = Input (shape = (input_dim))
    # =========================================================================
    act_ = fcrn_nalu (input_)
    # =========================================================================
    density_pred =  Convolution2D(1, 1, 1, bias = False, activation='linear',\
                                  init='orthogonal',name='pred',border_mode='same')(act_)
    # =========================================================================
    model = Model (input = input_, output = density_pred)
    opt = SGD(lr = 1e-2, momentum = 0.9, nesterov = True)
    model.compile(optimizer = opt, loss = 'mse')
    return model

def buildmodel_u_net_nalu (input_dim):
    input_ = Input (shape = (input_dim))
    # =========================================================================
    act_ = u_net_nalu (input_, nb_filter = 64 )
    # =========================================================================
    density_pred =  Convolution2D(1, 1, 1, bias = False, activation='linear',\
                                  init='orthogonal',name='pred',border_mode='same')(act_)
    # =========================================================================
    model = Model (input = input_, output = density_pred)
    opt = SGD(lr = 1e-2, momentum = 0.9, nesterov = True)
    model.compile(optimizer = opt, loss = 'mse')
    return model
