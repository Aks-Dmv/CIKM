from clusterWorld import *
from anytree import AnyNode, RenderTree
from anytree.exporter import DotExporter
#import graphviz
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
from replay_buffer import ReplayBuffer
from Qnetwork import *

# List of hyper-parameters and constants
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 2
TOT_FRAME = 1000000
EPSILON_DECAY = 300000
MIN_OBSERVATION = 50
FINAL_EPSILON = 0.1
INITIAL_EPSILON = 1.0
TREE_DEPTH=5

class agent():

    ActionStatePairs=[]

    def __init__(self):
        self.ActionStatePairs=[]
        self.env=gameEnv()
        self.root=AnyNode(name="root",inheritedN=0)
        self.PrevStepInheritedN=0
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.deep_q = Qnetwork()
        self.process_buffer = []
        self.process_buffer.append(self.env.initState())
        #print("procBuf",self.process_buffer)


    def takeAction(self,dim,val,stop,N,upDown):
        #print(self.env.step(dim,val,stop,N,upDown))
        bounds,reward,self.PrevStepInheritedN=self.env.step(dim,val,stop,N,upDown)
        #print(info,bounds)

    def updatePartition(self,dim,val,obs):
        self.ActionStatePairs.append([dim,val,obs])

    def load_network(self, path):
        self.deep_q.load_network(path)

    def convert_process_buffer(self):
        #print("process_buffer is ",self.process_buffer)
        LEle=self.process_buffer[0][-1]
        #print("last ele is",LEle)
        PBLen=len(self.process_buffer[0])
        #print("size before is", len(self.process_buffer[0]),self.process_buffer)
        appLen=max(0,TREE_DEPTH-PBLen)
        print(self.process_buffer)
        self.addObservationNtimes(LEle,appLen)
        #print("process_buffer is ",self.process_buffer)
        #print("size is", len(self.process_buffer[0]),self.process_buffer)
        if(len(self.process_buffer[0])!=TREE_DEPTH):
            print(appLen,"this was applen")
            print()
            print("PBLen is ",PBLen)
            print(len(self.process_buffer[0]),TREE_DEPTH)
            print(self.process_buffer[0])
            print("################################################")
            print("process_buffer")
            print(self.process_buffer)
            print(LEle)
            return 0
        return np.array(self.process_buffer),LEle


    def getEnvInputs(self, opt_Pol,upD, regr,n):
        # print("optPolicy",upDown,upDown[0])
        opt_policy=opt_Pol[0][0]
        if(opt_policy<n):
            dim=opt_policy
            stop=0
        else:
            dim=0 # Some random value
            stop=1

        val=regr[0]
        inheritedN=self.env.childNodeN(self.PrevStepInheritedN)
        return dim,val,stop,inheritedN,upD[0]

    def addObservation(self,temp_observation):
        if(len(self.process_buffer[0])>=TREE_DEPTH):
            print("len >TREE_DEPTH")
            return
        temp=self.process_buffer[0]
        temp=temp.tolist()
        temp.append(temp_observation[0])
        temp=np.array(temp)

        new_process_buffer=[]
        new_process_buffer.append(temp)
        self.process_buffer=new_process_buffer

    def addObservationNtimes(self,temp_observation,n):
        temp=self.process_buffer[0]
        temp=temp.tolist()
        for i in range(n):
            temp.append(temp_observation)
        temp=np.array(temp)
        if(len(temp)>10):
            print("len >10",len(temp),temp)
        new_process_buffer=[]
        new_process_buffer.append(temp)
        self.process_buffer=new_process_buffer

    def train(self, num_frames):
        observation_num = 0

        #print("curr State",curr_state)
        epsilon = INITIAL_EPSILON
        alive_frame = 0
        total_reward = 0
        done=False
        #print("process buff",self.process_buffer)
        while observation_num < num_frames:
            if observation_num % 1000 == 999:
                print(("Executing loop %d" %observation_num))
                self.env.renderEnv()

            # Slowly decay the learning rate
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

            if done or alive_frame>TREE_DEPTH:


                # The above will always pad to length ten

                #EpisodeTrain=np.lib.pad(EpisodeTrain, (0,lim), 'constant', constant_values=(0, [0,0,0,0]))
                print("done is",done)
                curr_state, last_element = self.convert_process_buffer()
                last_element=np.array(last_element)
                last_element=last_element.reshape(1,1,4)
                print("next_state",last_element,last_element.shape)
                predict_movement, predict_q_value, regressor, upDown = self.deep_q.predict_movement(curr_state, epsilon)
                self.replay_buffer.add(curr_state, predict_movement,upDown,regressor, total_reward, done, last_element)
                self.process_buffer=[]
                self.process_buffer.append(self.env.initState())
                curr_state = np.array(self.process_buffer)
                print("Lived with maximum time ", alive_frame)
                print("Earned a total of reward equal to ", total_reward)
                #self.env.renderEnv()
                self.env.reset()
                alive_frame = 0
                total_reward = 0
                done=False
                continue

            curr_state = np.array(self.process_buffer)
            print("curr State outside if",curr_state,curr_state.shape)
            #self.process_buffer = []

            predict_movement, predict_q_value, regressor, upDown = self.deep_q.predict_movement(curr_state, epsilon)

            dim,val,stop,inheritedN,upDown=self.getEnvInputs(predict_movement,upDown,regressor,self.deep_q.NUM_DIM)

            # reward, done = 0, False
            #print("hi")
            #print("hello",dim,val,stop,inheritedN,upDown)
            #print(self.env.step(dim,val,stop,inheritedN,upDown))
            temp_observation, temp_reward, temp_done, NewN = self.env.step(dim,val,stop,inheritedN,upDown)
            total_reward += temp_reward
            #print("what we are adding",temp_observation,curr_state,curr_state[0],curr_state[0][0])
            print("I am an ass",self.process_buffer)
            self.addObservation(temp_observation)

            done = done | bool(temp_done)

            if observation_num % 10 == 0:
                print("We predicted a q value of ", predict_q_value)







            if self.replay_buffer.size() > MIN_OBSERVATION:
                s_batch, a_batch,upDown_batch,reg_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(MINIBATCH_SIZE)
                self.deep_q.train(s_batch, a_batch,upDown_batch,reg_batch, r_batch, d_batch, s2_batch, observation_num)
                self.deep_q.target_train()

            # Save the network every 100000 iterations
            if observation_num % 10000 == 9999:
                print("Saving Network")
                self.deep_q.save_network("saved.h5")

            alive_frame += 1
            observation_num += 1



    def takeTempPolicy(self):
        #li(dim,val,upDown,stop)
        li=[[0,0,1,0],[0,2,1,0],[1,2,1,0],[1,3,1,0],[0,3,0,0],[1,4,1,0],[0,4,0,1]]
        if(len(li)==0):
            return
        nodes= [AnyNode(name="temp",parent=self.root,dim=0,part=0,inheritedN=0) for i in range(len(li))]
        parentNode=self.root
        TreeTravDirec=li[0][0]
        #print(TreeTravDirec)
        for i in range(len(li)):
            if(li[i][0] != TreeTravDirec):
                TreeTravDirec=li[i][0]
                parentNode=nodes[i-1]
            inheritedN=self.env.childNodeN(self.PrevStepInheritedN)
            nodes[i].name = str(i)
            nodes[i].dim = str(li[i][0])
            nodes[i].val = str(li[i][1])
            nodes[i].parent = parentNode
            nodes[i].inheritedN = inheritedN

            self.takeAction(li[i][0],li[i][1],li[i][3],inheritedN,li[i][2])

        self.env.renderEnv()
            #print(nameNode,"yo")
            #AnyNode(name="sub0B", parent=s0, index=1, partition=9)
        #DotExporter(self.root).to_picture("TreeDiag/root.png")





if __name__== "__main__":
    a=agent()
    a.train(100000)
    #print(a.process_buffer)
