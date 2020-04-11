# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:39:23 2020

@author: Heng Fu

Title: First Attempt For Joint Pilot Design and Channel Estimation in OFDM System
"""
import tensorflow as tf
import numpy as np

###############################################################################
## Load training and testing dataset
channel_train = np.load('channel_train.npy')
train_size = channel_train.shape[0]
print("\nTrain_size:",train_size)

channel_test = np.load('channel_test.npy')
test_size = channel_test.shape[0]
print("\ntest_size:",test_size)

##############################################################################
## Neural network parameters

n_input    =   32;
n_hidden_1 =  500;
n_hidden_2 =  250;
n_hidden_3 =  125;
n_hidden_4 =   32;

S_dim      =    8;

n_hidden_5 = 1000;
n_hidden_6 =  500;
n_hidden_7 =  250;
n_hidden_8 =  125;
n_hidden_9 =   32;


###############################################################################
## Customize Top_S Layer for selecting the positions of the largest k probability, and set 
## the corresponding positions to 1. The remaining positions will be set to 0.   

def top_n_filter_layer(input_data, S_dim=8): ## Use it in tf.keras.layers.Lambda()
    
    topk = tf.nn.top_k(input_data, k=S_dim, sorted=False)
    
    #  
    res  = tf.reduce_sum(tf.one_hot(topk.indices, input_data.get_shape().as_list()[-1]), axis = 1)
      
    return res

###############################################################################
## Customize my own Top_S layer    
class TopS_Layer(tf.keras.layers.Layer):
    
    def __init__(self, dimension_S, dimension):
        super(TopS_Layer, self).__init__()
        self.dimension_S = dimension_S
        self.output_result = tf.Variable(initial_value = tf.zeros((dimension,)),
                                  trainable = False)
  
    def call(self, inputs):        
        self.output_result = tf.reduce_sum(tf.one_hot(tf.nn.top_k(inputs, k = self.dimension_S, sorted=False).indices, 
                                                      inputs.get_shape().as_list()[-1]), axis = 1)
        return self.output_result    


## Instantiating the customized layer
Tops_Layer = TopS_Layer(S_dim, n_input)


###############################################################################
def Xor_Operation(input_data, tops_output):
    
    return input_data * tops_output


###############################################################################
def encoder_process(h_sample):
    # input_channel_sample = tf.keras.Input(shape=(n_input,))
    temp = tf.keras.layers.Dense(n_hidden_1, activation='relu')(h_sample)
    temp = tf.keras.layers.Dense(n_hidden_2, activation='relu')(temp)
    temp = tf.keras.layers.Dense(n_hidden_3, activation='relu')(temp)
    temp = tf.keras.layers.Dense(n_hidden_4, activation=tf.nn.softmax)(temp)
    top_s_vector = tf.keras.layers.Lambda(top_n_filter_layer, output_shape=(S_dim,))(temp)
    #top_s_vector = Tops_Layer(temp)   
    return top_s_vector
    #return temp


#####################################    ##########################################
def decoder_process(encoded_vector):
    # input_vector = tf.keras.Input(shape=(S_dim,))
    temp = tf.keras.layers.Dense(n_hidden_5, activation='relu')(encoded_vector)
    temp = tf.keras.layers.Dense(n_hidden_6, activation='relu')(temp)
    temp = tf.keras.layers.Dense(n_hidden_7, activation='relu')(temp)
    temp = tf.keras.layers.Dense(n_hidden_8, activation='relu')(temp)
    output_vector = tf.keras.layers.Dense(n_hidden_9)(temp)
    return output_vector


###############################################################################
def create_model(h_sample):
   
    # The encoder first obtains the sparse binary vector 
    encoded_vector = encoder_process(h_sample)
    
    # encoded_vector goes through XOR operation
    processed_vector =  tf.keras.layers.Lambda(top_n_filter_layer)(h_sample, encoded_vector) 
    
    # Decoder can decode the code
    output = decoder_process(processed_vector)
    #output = decoder_process(encoded_vector)
    
    return output
    
###############################################################################    
input_tensor  = tf.keras.layers.Input(shape=(n_input, ))  
output_tensor = create_model(input_tensor)    
Final_Model = tf.keras.models.Model(inputs=[input_tensor], outputs=[output_tensor])

    
Final_Model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])

Final_Model.summary()

Result = Final_Model.predict(tf.random.normal(shape = (32, 32)))
print("\nNetwork_Predict[0]:", Result)
    
## Training Data                   
def training_gen(sample_size):                                              
    while True:
        index = np.random.choice(np.arange(train_size), size=sample_size)      # 从arrange(train_size)中随便选取bs个数构成一个列表
        H_total = channel_train[index]                                         # 从channel_train中挑选这些对应索引的信道值 
        input_samples = []          
        input_labels = []
        
        for H in H_total:            
            input_samples.append(np.concatenate((np.real(H), np.imag(H))))                                
            input_labels.append(np.concatenate((np.real(H), np.imag(H))))          
        yield (np.asarray(input_samples), np.asarray(input_labels))    
   
    
## Testing Data      
def validation_gen(sample_size):
    while True:
        index = np.random.choice(np.arange(test_size), size=sample_size)
        H_total = channel_test[index]
        input_samples = []
        input_labels = []
        
        for H in H_total:
            input_samples.append(np.concatenate((np.real(H), np.imag(H))))
            input_labels.append(np.concatenate((np.real(H), np.imag(H))))
            
        yield (np.asarray(input_samples), np.asarray(input_labels))    

## Train the Model
history = Final_Model.fit(training_gen(10000),
                steps_per_epoch = 100,
                epochs = 10,
                validation_data = validation_gen(10000),
                validation_steps = 1,
                verbose = 1)
    
    
    
    
    
    
    
    
    
    