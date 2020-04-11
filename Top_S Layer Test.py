# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:36:38 2020

@author: Heng Fu
"""

import tensorflow as tf
#import numpy as np

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer((10,)))

def top_n_filter_layer(input_data, n=2):
    
    topk = tf.nn.top_k(input_data, k=n, sorted=False)
    
    res  = tf.reduce_sum(tf.one_hot(topk.indices, input_data.get_shape().as_list()[-1]), axis = 1)
    
    res  *= input_data 
    
    return res
    
model.add(tf.keras.layers.Lambda(top_n_filter_layer))

model.summary()

x_train = tf.Variable([[1., 2., 3., 4., 5., 6., 7., 7., 7., 7.], [2., 3., 5., 8., 1., 2., 5., 4., 0., 9.]])

result = top_n_filter_layer(x_train)

result = model.predict((x_train))
print("\nResult:", result)

print("\n####################################################################1")
###############################################################################
## Test
topk = tf.math.top_k(x_train, k=2, sorted = False)
print("\nTop_k:", topk)

print("\nIndices:", topk.indices)
print("\nValues:" , topk.values )

onehot = tf.one_hot(topk.indices, x_train.get_shape().as_list()[-1])
print("\nonehot_code:", onehot)

res = tf.reduce_sum(onehot, axis = 1)
print("\nres:", res)

res = res* x_train
print("\nres:", res)

print("\n####################################################################2")

############################################################################### 
## Another method

import tensorflow as tf
# import numpy as np

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10, input_shape=(5,),
                                activation = tf.nn.softmax,
                                kernel_initializer = tf.random_normal_initializer))

def top_n_filter_layer(input_data, n=2):
    
    topk = tf.nn.top_k(input_data, k=n, sorted=False)
    
    res  = tf.reduce_sum(tf.one_hot(topk.indices, input_data.get_shape().as_list()[-1]), axis = 1)
    
    res  *= input_data 
    
    return res

model.add(tf.keras.layers.Lambda(top_n_filter_layer))

model.summary()

x_train = tf.Variable(tf.random.normal((2, 5)))
print("\nx_train:", x_train)

result = model.predict(x_train)
print("\nresult:", result)
print("\n####################################################################3")

##############################################################################
## Outputing the one-hot codes can meet our requirement

import tensorflow as tf
# import numpy as np

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10, input_shape=(5,),
                                activation = tf.nn.softmax,
                                kernel_initializer = tf.random_normal_initializer))

def top_n_filter_layer(input_data, n=2):
    
    topk = tf.nn.top_k(input_data, k=n, sorted=False)
    
    # Set the maximum positions to 1 by one-hot encoding, and the remaining positions to 0 
    res  = tf.reduce_sum(tf.one_hot(topk.indices, input_data.get_shape().as_list()[-1]), axis = 1)
      
    return res

model.add(tf.keras.layers.Lambda(top_n_filter_layer))

model.summary()

x_train = tf.Variable(tf.random.normal((2, 5)))
print("\nx_train:", x_train)

result = model.predict(x_train)
print("\nresult:", result)

print("\n####################################################################4")

##############################################################################
## Element-wise multiplication

def Xor_Operation(input_data, tops_output):
    
    return input_data * tops_output

input_data  = tf.constant([[1., 2., 3., 4.],[6., 7., 4., 5.]])
tops_output = tf.constant([[0., 0., 1., 1.], [1., 1., 0., 0.]])

Result = Xor_Operation(input_data, tops_output)
print("\nResult；", Result)
    


























