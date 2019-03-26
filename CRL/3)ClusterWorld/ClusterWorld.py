import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import copy
from infoCalc import *


class ClusterWorldEnv:
    """
    Define a simple ClusterWorld environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """


    def __init__(self):
        # General variables defining the environment
        self._trueDf=pd.read_csv("../data/dataPts.csv")
        self._trueBoundaries=np.array([[-1.,4.],[1.,6.]])


        # Our ouput should be in the form of
        # (N output variables, one stop variable) and one regressor variable
        SoftMaxOutput = np.tile( [0.,1.], (len(self._trueBoundaries)+1,1) )
        Regr=np.array([[self._trueBoundaries.min(), self._trueBoundaries.max()]])
        self.action_space = np.concatenate((SoftMaxOutput, Regr), axis=0)

        ob = copy.deepcopy(self._trueBoundaries)
        self.observation_space = np.reshape(ob,(1,4))[0]

        self.observation_space_Matrix=copy.deepcopy(self._trueBoundaries)
        self.actionstatepairs=[]
        self.df=copy.deepcopy(self._trueDf)
        self.inheritedN=0
        self._infoRewardMultiplier=1000
        self.penalty=-1
        self._Quitpenalty=-10
        self.reward=0
        self.done=False
        self._maxRegressVal=10

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.observation_space_Matrix=copy.deepcopy(self._trueBoundaries)
        self.actionstatepairs=[]
        self.df=copy.deepcopy(self._trueDf)
        self.inheritedN=0
        self.penalty=-1
        self.reward=0
        self.done=False
        return self._get_state()

    def _get_state(self):
        """Get the observation."""
        ob = copy.deepcopy(self.observation_space_Matrix)
        #print(ob)
        ob = np.reshape(ob,(1,4))[0]
        return ob

    def seed(self, seed):
        random.seed(seed)
        np.random.seed

    def _normConstr(self,val,dim):
        #the tempDim value has been hard coded
        # because of the dataset
        tempDim=1-dim
        dimRange=self._trueBoundaries[tempDim][1]-self._trueBoundaries[tempDim][0]
        valNew=(val-self._trueBoundaries[tempDim][0])/dimRange

        return valNew

    def render(self):
        plt.scatter(self._trueDf['0'], self._trueDf['1'])
        if(len(self.actionstatepairs)==0):
            #print("empty list")
            return
        prevI=self.actionstatepairs[0]
        Max_Render_space=copy.deepcopy(self._trueBoundaries)
        for i in self.actionstatepairs:
            #print(i)
            if(i[0]==1):
                plt.axhline(y=i[1], xmin=self._normConstr(Max_Render_space[0][0],i[0]), xmax=self._normConstr(Max_Render_space[0][1],i[0]), color='r', linestyle='-')

            else:
                plt.axvline(x=i[1], ymin=self._normConstr(Max_Render_space[1][0],i[0]), ymax=self._normConstr(Max_Render_space[1][1],i[0]), color='r', linestyle='-')

            # The 1-i[2] just means that we are setting the min
            # to the val ie (i[1]) if we chose to go up
            Max_Render_space[i[0]][1-i[2]]=i[1]
        plt.title('Scatter plot pythonspot.com')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show(block=False)
        input("enter")
        plt.close()

    def updateState(self):
        i=copy.deepcopy(self.actionstatepairs[-1])

        self.observation_space_Matrix[int(i[0])][1-int(i[2])]=i[1]



    def stepThroughTree(self,action,Up):

        softmaxOut=action[:-1].argmax()
        regrOut=action[-1]
        if(softmaxOut==len(self._trueBoundaries) or self.done):
            # If you are here, then the stop variable has been flagged
            self.done=True
            bounds=self._get_state()
            self.reward=self._Quitpenalty #/(len(self.actionstatepairs)+1)
            return bounds, self.reward,self.done,None

        # If we have not returned yet, then that means the softmaxOut represents which dim
        dim=softmaxOut
        actionMultiFactor=(self.observation_space_Matrix[dim][1]-self.observation_space_Matrix[dim][0])/self._maxRegressVal
        val=regrOut*actionMultiFactor+self.observation_space_Matrix[dim][0]
        #print("1val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1]",val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1])

        # to check if the regressor is out of bounds
        if(val<=self.observation_space_Matrix[dim][0]):
            lessTh=True
        else:
            lessTh=False
        if(val>=self.observation_space_Matrix[dim][1]):
            greaterTh=True
        else:
            greaterTh=False
        if(lessTh or greaterTh):
            # you exceeded the boundary
            # this includes selecting the boundary
            # Thus, your action was wrong
            # Self.done is false because you just picked a wrong value
            # you can try again
            # note that the environment will not store this value
            # however, the S,A,R,S will be stored by the agent
            self.done=False
            bounds=self._get_state()
            if(lessTh):
                deltaBounds=self.observation_space_Matrix[dim][0]-val

            if(greaterTh):
                deltaBounds=val-self.observation_space_Matrix[dim][1]
            self.reward=self._Quitpenalty*(abs(deltaBounds))
            return bounds, self.reward,self.done,None

        #print("2val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1]",val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1])
        #Up= int(np.random.random(1)>0.5)
        #Up=1

        # Checking for a corner case
        if(len(self.df.index)==0):
            # There are no elements
            # Thus, your action was futile
            self.done=True
            bounds=self._get_state()
            self.reward=-1
            return bounds, self.reward,self.done,None



        #print("3val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1]",val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1])
        InfoGain,df1,df2,N1,N2=infoGain(self.df,dim,val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1],self.inheritedN)

        self.reward = self._infoRewardMultiplier*InfoGain-self.penalty


        if(Up==1):
            self.df=df1
            self.inheritedN=N1
        else:
            self.df=df2
            self.inheritedN=N2

        self.actionstatepairs.append([int(dim),val,int(Up)])
        #print("updatingState")
        self.updateState()

        bounds=self._get_state()

        return bounds, self.reward,self.done,None


    def step(self,action):

        softmaxOut=action[:-1].argmax()
        regrOut=action[-1]
        if(softmaxOut==len(self._trueBoundaries) or self.done):
            # If you are here, then the stop variable has been flagged
            self.done=True
            bounds=self._get_state()
            self.reward=self._Quitpenalty #/(len(self.actionstatepairs)+1)
            return bounds, self.reward,self.done,None

        # If we have not returned yet, then that means the softmaxOut represents which dim
        dim=softmaxOut
        actionMultiFactor=(self.observation_space_Matrix[dim][1]-self.observation_space_Matrix[dim][0])/self._maxRegressVal
        val=regrOut*actionMultiFactor+self.observation_space_Matrix[dim][0]
        #print("1val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1]",val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1])

        # to check if the regressor is out of bounds
        if(val<=self.observation_space_Matrix[dim][0]):
            lessTh=True
        else:
            lessTh=False
        if(val>=self.observation_space_Matrix[dim][1]):
            greaterTh=True
        else:
            greaterTh=False
        if(lessTh or greaterTh):
            # you exceeded the boundary
            # this includes selecting the boundary
            # Thus, your action was wrong
            # Self.done is false because you just picked a wrong value
            # you can try again
            # note that the environment will not store this value
            # however, the S,A,R,S will be stored by the agent
            self.done=False
            bounds=self._get_state()
            if(lessTh):
                deltaBounds=self.observation_space_Matrix[dim][0]-val

            if(greaterTh):
                deltaBounds=val-self.observation_space_Matrix[dim][1]
            self.reward=self._Quitpenalty*(abs(deltaBounds))
            return bounds, self.reward,self.done,None

        #print("2val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1]",val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1])
        Up= int(np.random.random(1)>0.5)
        #Up=1

        # Checking for a corner case
        if(len(self.df.index)==0):
            # There are no elements
            # Thus, your action was futile
            self.done=True
            bounds=self._get_state()
            self.reward=-1
            return bounds, self.reward,self.done,None



        #print("3val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1]",val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1])
        InfoGain,df1,df2,N1,N2=infoGain(self.df,dim,val,self.observation_space_Matrix[dim][0],self.observation_space_Matrix[dim][1],self.inheritedN)

        self.reward = self._infoRewardMultiplier*InfoGain-self.penalty


        if(Up==1):
            self.df=df1
            self.inheritedN=N1
        else:
            self.df=df2
            self.inheritedN=N2

        self.actionstatepairs.append([int(dim),val,int(Up)])
        #print("updatingState")
        self.updateState()

        bounds=self._get_state()

        return bounds, self.reward,self.done,None
