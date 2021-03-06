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


class BananaEnv:
    """
    Define a simple Banana environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """


    def __init__(self):
        # General variables defining the environment
        self._trueDf=pd.read_csv("dataPts.csv")
        self._trueBoundaries=np.array([[-1,4],[1,6]])


        # Our ouput should be in the form of
        # (N output variables, one stop variable) and one regressor variable
        SoftMaxOutput = np.tile( [0,1], (len(self._trueBoundaries)+1,1) )
        Regr=np.array([[self._trueBoundaries.min(), self._trueBoundaries.max()]])
        self.action_space = np.concatenate((SoftMaxOutput, Regr), axis=0)


        self.observation_space=copy.deepcopy(self._trueBoundaries)
        self.actionStatePairs=[]
        self.df=copy.deepcopy(self._trueDf)
        self.inheritedN=0
        self.infoRewardMultiplier=100
        self.penalty=0
        self.reward=0
        self.done=False

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.observation_space=copy.deepcopy(self._trueBoundaries)
        self.actionStatePairs=[]
        self.df=copy.deepcopy(self._trueDf)
        self.inheritedN=0
        self.penalty=0
        self.reward=0
        self.done=False
        return self._get_state()

    def _get_state(self):
        """Get the observation."""
        ob = copy.deepcopy(self.observation_space)
        return ob

    def seed(self, seed):
        random.seed(seed)
        np.random.seed

    def _normConstr(self,val,dim):
        #the tempDim value has been hard coded
        # because of the dataset
        tempDim=1-dim
        dimRange=TRUE_BOUNDARIES[tempDim][1]-TRUE_BOUNDARIES[tempDim][0]
        valNew=(val-TRUE_BOUNDARIES[tempDim][0])/dimRange

        return valNew

    def render(self):
        plt.scatter(TRUE_DF['0'], TRUE_DF['1'])
        if(len(self.actionstatepairs)==0):
            #print("empty list")
            return
        prevI=self.actionstatepairs[0]
        Max_Render_space=copy.deepcopy(TRUE_BOUNDARIES)
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
        i=self.actionstatepairs[-1]
        self.observation_space[int(i[0])][1-int(i[2])]=int(i[1])


    def step(self,action):

        softmaxOut=action[:-1].argmax()
        regrOut=action[-1]
        if(softmaxOut==len(self._trueBoundaries) or self.done):
            # If you are here, then the stop variable has been flagged
            self.done=True
            bounds=self._get_state(self.observation_space)
            NewN=inheritedN
            self.reward=0
            return bounds, self.reward,self.done,None

        # If we have not returned yet, then that means the softmaxOut represents which dim
        dim=softmaxOut
        actionMultiFactor=(self.observation_space[dim][1]-self.observation_space[dim][0])
        val=regrOut*actionMultiFactor+self.observation_space[dim][0]

        if(val<=self.observation_space[dim][0] or val>=self.observation_space[dim][1]):
            # you exceeded the boundary
            # this includes selecting the boundary
            # Thus, your action was wrong
            # Self.done is false because you just picked a wrong value
            # you can try again
            # note that the environment will not store this value
            # however, the S,A,R,S will be stored by the agent
            self.done=False
            bounds=self._get_state(self.observation_space)
            NewN=inheritedN
            self.reward=-10
            return bounds, self.reward,self.done,None


        Up= int(np.random.random(1)>0.5)
        self.actionstatepairs.append([int(dim),int(val),int(Up)])
        self.updateState()

        # Checking for a corner case
        if(len(self.df.index)==0):
            # There are no elements
            # Thus, your action was futile
            self.done=True
            bounds=self._get_state(self.observation_space)
            NewN=inheritedN
            self.reward=0
            return bounds, self.reward,self.done,None




        InfoGain,df1,df2,N1,N2=_infoGain(self.df,dim,val,self.observation_space[dim][0],self.observation_space[dim][1],self.inheritedN)

        self.reward = self.infoRewardMultiplier*InfoGain-self.penalty


        if(Up==1):
            self.df=df1
            self.inheritedN=N1
        else:
            self.df=df2
            self.inheritedN=N2

        bounds=self._get_state(self.observation_space)

        return bounds, self.reward,self.done,None
