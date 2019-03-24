import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input, Subtract, Average,Add
from keras.models import Model
from keras.backend import repeat_elements,mean

sess=tf.Session()
K.set_session(sess)

def nn(input_Shape):
    XInput=Input(input_Shape)
    X1= Dense(16,activation='softmax')(XInput)
    X2= Dense(16,activation='softmax')(XInput)
    X1_mean=Add()([X1,X2])
    #Y=Input(X1_mean)
    #X1_Av_Tensor=repeat_elements(X1_mean)
    #Y=Dense(16,activation='softmax')(X1_Av_Tensor)
    #OrigShift=Subtract()
    #Y=self.Value + tf.subtract(X1,tf.reduce_mean(X1,axis=1,keepdims=True))
    return Model(XInput,X1_mean)

model=nn((32,5))
model.summary()
