import numpy as np
import tensorflow as tf
import random
from keras import backend as K
from keras.layers import Dense, Input, Subtract, Activation
from keras.layers import Lambda, LSTM, Dropout, Concatenate
from keras.models import Model
from keras.backend import repeat_elements,mean
from keras.optimizers import Adam


# we are setting NUM_ACTIONS to be 2 (2 dim hyperspace) + 1 done/stop


class Qnetwork():
    def policyFn(self,x):
        return x[0]-K.mean(x[0])+x[1]

    def NetworkSummary(self):
        self.target_model.summary()
        print("the last layer is like [regressor,dim0,dim1,stop,UpDown]")

    def __init__(self):
        self.DECAY_RATE = 0.99
        self.NUM_DIM=2
        self.NUM_ACTIONS = self.NUM_DIM+2
        self.TREE_DEPTH=10

        self.model = self.DRQN()
        self.model.compile(loss='mse', optimizer=Adam(lr=0.000001))

        self.target_model = self.DRQN()
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.000001))
        print("Successfully constructed both networks.")

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
        print("Succesfully loaded network.")

    def target_train(self):
        model_weights = self.model.get_weights()
        self.target_model.set_weights(model_weights)


    def DRQN(self):

        # Here, we take in an input and pass it to a two layer RNN
        # Which is followed by a two dense layers
        XInput=Input([None,2*self.NUM_DIM])

        # X1=LSTM(2*self.NUM_ACTIONS, return_sequences=True)(XInput)
        # X1=LSTM(2*self.NUM_ACTIONS, return_sequences=False)(X1)

        X1=LSTM(2*self.NUM_ACTIONS, return_sequences=False)(XInput)
        X1=Dropout(0.3)(X1)
        X1= Dense(2*self.NUM_ACTIONS,activation='relu')(X1)
        X1=Dropout(0.5)(X1)

        # OK, now we have to get the advantage and value functions

        Adv=Dense(self.NUM_ACTIONS,activation='relu')(X1)
        Val=Dense(self.NUM_ACTIONS,activation='relu')(X1)

        # Now, we just have to find the Q-value
        X2 = Lambda(self.policyFn, output_shape = (self.NUM_ACTIONS,))([Adv,Val])
        policy=Activation('softmax')(X2)
        # This will give the regressor
        regressor= Dense(1,activation='relu')(X2)

        UpDown= Dense(1,activation='relu')(X2)
        UpDown=Activation('softmax')(UpDown)

        finalOutput=Concatenate()([regressor,policy,UpDown])

        return Model(XInput,finalOutput)


    def predict_movement(self, data, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        siz=data.shape[0]
        #print(siz)
        tempOutput = self.target_model.predict(data, batch_size = siz)
        #print(tempOutput)
        regress,q_actions, UpDown= self.getRegressDim(tempOutput)
        #print("regressor",regress)
        #print("q_actions",q_actions)
        opt_policy=[]
        opt_policy.append(np.argmax(q_actions,axis=1))
        opt_policy=np.array(opt_policy)
        #print("hello")
        #print(opt_policy)
        #print("opt policy",opt_policy)
        rand_val = np.random.random()
        if rand_val < epsilon:
            #print("if running")
            opt_policy = np.random.randint(self.NUM_ACTIONS,size=(1,siz))
        #print(opt_policy)
        #print(opt_policy)
        indices=np.array(np.eye(self.NUM_ACTIONS)[opt_policy].astype(bool))
        #print(opt_policy)
        #print(indices)
        #print(indices.shape,indices,q_actions.shape)
        #print("the return")
        #print("youoa")
        #print(indices[0,:])
        #print(q_actions)

        #print(q_actions[indices[0,:]])
        return opt_policy, q_actions[indices[0,:]], regress,UpDown

    def getRegressDim(self,temp):
        # This function removes the regressor aspect
        # print(temp.shape)
        # print(temp[:,0])
        # print(temp[0,1:])
        # print(temp[:,-1])
        return temp[:,0],temp[:,1:-1],temp[:,-1]


    def train(self, s_batch, a_batch,upDown_batch,reg_batch, r_batch,d_batch, s2_batch, observation_num):
        #Setting the training parameters
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, self.NUM_ACTIONS+2))

        for i in range(batch_size):
            #print("targets",targets[i])
            #print("s_batch",s_batch)
            #print("model predict",self.model.predict(s_batch[i], batch_size = 1)[0])

            targets[i] = self.model.predict(s_batch[i], batch_size = 1)
            # print(targets[i])
            # print("targets shape",targets[i].shape)
            # print(s2_batch[i])

            fut_action = self.target_model.predict(s2_batch[i], batch_size = 1)
            # print(fut_action)
            # print(fut_action.shape)
            # fut_action=np.array(fut_action)
            # print(fut_action)
            # print(fut_action.shape)
            Futregress,Futq_actions, FutUpDown= self.getRegressDim(fut_action)
            targets[i, 0] = r_batch[i]
            targets[i, a_batch[i]+1] = r_batch[i]
            targets[i, -1] = r_batch[i]

            if d_batch[i] == False:
                targets[i, 0] += self.DECAY_RATE * Futregress
                targets[i, a_batch[i]] += self.DECAY_RATE * np.max(Futq_actions)
                targets[i, -1] += self.DECAY_RATE * FutUpDown

        #print(s_batch)
        #print("before",s_batch.shape)
        s_batch=s_batch.squeeze()
        #print("after",s_batch.shape)
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



#######Uncomment Here
# tf.logging.set_verbosity(tf.logging.ERROR)
# sess=tf.Session()
# K.set_session(sess)
#
#
# #the input is of shape timesteps x
#
# model=Qnetwork()
# model.NetworkSummary()
# data=[[[1,2,1,5],[1,4,1,3],[2,2,3,1],[1,3,2,1],[1,3,1,2],[2,4,1,4],[1,3,1,2],[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[1,2,1,3],[1,2,1,3],[1,2,1,1],[1,3,1,3],[1,3,3,3],[1,4,1,4],[2,4,1,1],[1,2,3,4],[1,2,3,4],[1,2,3,4]]]
# #data=[[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]]
# data=np.array(data)
# print(data.shape)
# ans=model.predict_movement(data,1)
# print(ans)
