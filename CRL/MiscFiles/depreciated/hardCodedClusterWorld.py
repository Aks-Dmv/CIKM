import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import copy

"""
This file will emulate our cluster world environment. The env has a specific set of tasks to do. They can be summed up as:

init - read a csv file and get the points
step - Takes a step in the env based on the partition
reward - returns the reward based on the action
reset
rendering
stop (or checkGoal)

Code to select rows from DataFrame
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
                   'B': 'one one two three two two one three'.split(),
                   'C': np.arange(8), 'D': np.arange(8) * 2})
print(df)
#      A      B  C   D
# 0  foo    one  0   0
# 1  bar    one  1   2
# 2  foo    two  2   4
# 3  bar  three  3   6
# 4  foo    two  4   8
# 5  bar    two  5  10
# 6  foo    one  6  12
# 7  foo  three  7  14

print(df.loc[df['A'] == 'foo'])

Steps to do:

First import a csv as a dataframe
make a function to calc info given a set of points

"""

class gameEnv:

    a=[0.6,0.51,0.6,0.6,0.3,0.6,0.3]
    Aindex=0

    def __init__(self):
        self.indexMinMax=[]
        self.OrigMinMax=[]
        self.indexMinMax.append([-1,4]) # for X
        self.indexMinMax.append([1,6]) # for Y
        self.OrigMinMax.append([-1,4]) # for X
        self.OrigMinMax.append([1,6]) # for Y
        self.prevObs=1
        self.ActionStatePairs=[]
        self.TotInfo=0
        self.df=pd.read_csv("dataPts.csv")
        self.Truedf=pd.read_csv("dataPts.csv")
        self.inheritedN=0


        self.reward=0
        self.totalReward=0
        self.done=False
        #this is used to init variables



    def initEnvBoundaries(self):
        self.indexMinMax=copy.deepcopy(self.OrigMinMax)


    def childNodeN(self,inheritedN):
        Y=len(self.df.index)
        if(Y==0):
            Y=0.00001
        if(inheritedN<=Y):
            # This can be defined to any real number
            c=1
            inheritedN=c*Y
        #print("Y inheritedN",Y,inheritedN)
        return inheritedN


    def info(self,df,inheritedN):
        #print("inherited N", inheritedN)
        Y=len(df.index)
        if(Y==0):
            Y=0.00001
        probOfY=Y/(inheritedN+Y)
        probOfN=1-probOfY

        #print("printing probOfY,inheritedN,Y, probOfN",probOfY,inheritedN,Y,probOfN)
        infoY = -1*math.log(probOfY)
        infoN = -1*math.log(probOfN)
        ExpectedInfo = probOfY*infoY + probOfN*infoN
        return ExpectedInfo

    def infoGain(self,df,dim,val,start,end,inheritedN):
        # class importance is an array that
        # gives us the linear combination coeff
        # This can be given during runtime
        #print("start end dim val",start, end, dim, val)
        df1=df.loc[df[str(dim)] > val]
        df2=df.loc[df[str(dim)] < val]
        ModD1=len(df1.index)
        ModD2=len(df2.index)

        ModD=ModD1+ModD2
        if(ModD==0):
            ModD=0.00001
        #print("inheritedN",inheritedN)
        N1=((end-val)*inheritedN)/(end-start)
        N2=((val-start)*inheritedN)/(end-start)
        N1=self.childNodeN(N1)
        N2=self.childNodeN(N2)
        #print("N1 and N2",N1,N2)
        #print("going to call info twice")
        #print("ModD is",ModD)
        deltaInfo = -1*( ModD1*self.info(df1,N1)+ModD2*self.info(df2,N2) )/ModD
        return deltaInfo

    def normConstr(self,val,dim):
        #the tempDim value has been hard coded
        # because of the dataset
        tempDim=1-dim
        dimRange=self.OrigMinMax[tempDim][1]-self.OrigMinMax[tempDim][0]
        #print(self.OrigMinMax[tempDim][0],self.indexMinMax[tempDim][0])
        valNew=(val-self.OrigMinMax[tempDim][0])/dimRange
        # if(dim==0):
        #     valNew=(val-(-1))/5
        # else:
        #     valNew=(val-(1))/5
        #print("hello", val, valNew)
        return valNew

    def updateBoundary(self):
        i=self.ActionStatePairs[-1]
        self.indexMinMax[i[0]][1-i[2]]=i[1]
        #print(i[1])
        # if(i[0]==0):
        #
        #     if(i[2]==0):
        #         self.indexMinMax[1][1]=i[1]
        #     else:
        #         #print(i)
        #         self.indexMinMax[1][0]=i[1]
        # else:
        #
        #     if(i[2]==0):
        #         self.indexMinMax[0][1]=i[1]
        #     else:
        #         #print(i)
        #         self.indexMinMax[0][0]=i[1]

    def renderEnv(self):
        plt.scatter(self.Truedf['0'], self.Truedf['1'])
        if(len(self.ActionStatePairs)==0):
            #print("empty list")
            return
        prevI=self.ActionStatePairs[0]
        self.initEnvBoundaries()
        for i in self.ActionStatePairs:
            #print(i)
            if(i[0]==1):
                #print(self.minX,self.normConstr(self.minX,0), self.normConstr(self.maxX,0))
                plt.axhline(y=i[1], xmin=self.normConstr(self.indexMinMax[0][0],i[0]), xmax=self.normConstr(self.indexMinMax[0][1],i[0]), color='r', linestyle='-')
                #print("hline done")
                # if(i[2]==0):
                #     self.indexMinMax[1][1]=i[1]
                # else:
                #     #print(i)
                #     self.indexMinMax[1][0]=i[1]
            else:
                #print("vline start")
                plt.axvline(x=i[1], ymin=self.normConstr(self.indexMinMax[1][0],i[0]), ymax=self.normConstr(self.indexMinMax[1][1],i[0]), color='r', linestyle='-')
                #print("vline done")
                # if(i[2]==0):
                #     self.indexMinMax[0][1]=i[1]
                # else:
                #     #print(i)
                #     self.indexMinMax[0][0]=i[1]
            self.indexMinMax[i[0]][1-i[2]]=i[1]
        plt.title('Scatter plot pythonspot.com')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show(block=False)
        input("enter")
        # time.sleep(2)
        plt.close()



    #here, we will update the partition with
    #[Action,State] pairs, Action=[dim,val]
    # def updatePartition(self,dim,val,obs):
    #     self.ActionStatePairs.append([dim,val,obs])

    def step(self,dim,val,stop,inheritedN):
        #print("inheritedN",inheritedN)
        # This will update our df such that we focus
        # on only the new set of points that are on one
        # side of the data set

        # 1)Uncomment when you want stochasticity
        # For reproducing results, comment out
        #a=np.random.rand(1)


        # 2)convert this to just a instead of self.a and self.Aindex
        #print("before if val",val)
        if(stop==1 or self.done):
            self.done=True
            return np.random.rand(1)
        if(self.Aindex<len(self.a) and self.a[self.Aindex]>0.5):

            #3) comment out for stochasticity
            self.Aindex=self.Aindex+1

            #print(a,[dim,val,1])
            #print(self.df[str(dim)])

            self.ActionStatePairs.append([dim,val,1])
            self.updateBoundary()
            #print("before infogain val",val)
            self.reward=-1*self.infoGain(self.df,dim,val,self.indexMinMax[dim][0],self.indexMinMax[dim][1],inheritedN)
            self.totalReward+=self.reward
            print(self.reward,self.totalReward)
            self.df=self.df.loc[self.df[str(dim)] > val]
            return 1
        else:

            #4)comment out for stochasticity
            self.Aindex=self.Aindex+1

            #print(a,[dim,val,0])
            #print(self.df[str(dim)])
            self.ActionStatePairs.append([dim,val,0])
            self.updateBoundary()
            self.reward=-1*self.infoGain(self.df,dim,val,self.indexMinMax[dim][0],self.indexMinMax[dim][1],inheritedN)
            self.totalReward+=self.reward
            print(self.reward,self.totalReward)
            self.df=self.df.loc[self.df[str(dim)] < val]
            return 0
