import numpy as np
import tensorflow as tf
import random
from keras import backend as K
from keras.layers import Dense, Input, Subtract, Activation
from keras.layers import Lambda, LSTM, Dropout, Concatenate
from keras.models import Model
from keras.backend import repeat_elements,mean
from keras.optimizers import Adam

DECAY_RATE = 0.99
# we are setting NUM_ACTIONS to be 2 (2 dim hyperspace) + 1 done/stop
NUM_ACTIONS = 2+1

class Qnetwork():
    def policyFn(self,x):
        return x[0]-K.mean(x[0])+x[1]

    def NetworkSummary(self):
        self.target_model.summary()

    def __init__(self,input_Shape):
        self.model = self.DRQN(input_Shape)
        self.model.compile(loss='mse', optimizer=Adam(lr=0.000001))

        self.target_model = self.DRQN(input_Shape)
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.000001))
        print("Successfully constructed both networks.")


    def DRQN(self,input_Shape):

        # Here, we take in an input and pass it to a two layer RNN
        # Which is followed by a two dense layers
        XInput=Input(input_Shape)
        X1=LSTM(2*NUM_ACTIONS, return_sequences=True)(XInput)
        X1=LSTM(2*NUM_ACTIONS, return_sequences=True)(X1)
        X1=Dropout(0.3)(X1)
        X2= Dense(2*NUM_ACTIONS,activation='relu')(X1)
        X2=Dropout(0.5)(X2)
        X2= Dense(2*NUM_ACTIONS,activation='relu')(X1)


        finalOutput=Concatenate(axis=1)([X1,X2])
        finalOutput=Dense(2*NUM_ACTIONS,activation='relu')(X1)

        return Model(XInput,finalOutput)


    def predict_movement(self, data, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""

        q_actions = self.model.predict(data, batch_size = 1)
        opt_policy = np.argmax(q_actions)
        rand_val = np.random.random()
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, NUM_ACTIONS)
        return opt_policy, q_actions[0, opt_policy]


    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains network to fit given parameters"""
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, NUM_ACTIONS))

        for i in range(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)

        loss = self.model.train_on_batch(s_batch, targets)

        # Print the loss every 10 iterations.
        if observation_num % 10 == 0:
            print("We had a loss equal to ", loss)

# predict = tf.argmax(Qout,1)
#
#
# targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
# actions = tf.placeholder(shape=[None],dtype=tf.int32)
# actions_onehot = tf.one_hot(actions,N,dtype=tf.float32)




tf.logging.set_verbosity(tf.logging.ERROR)
sess=tf.Session()
K.set_session(sess)


#the input is of shape timesteps x

model=Qnetwork([None,4])
model.NetworkSummary()
