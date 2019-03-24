from hardCodedClusterWorld import *
from anytree import AnyNode, RenderTree
from anytree.exporter import DotExporter
import graphviz
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim


class agent():

    ActionStatePairs=[]

    def __init__(self):
        self.ActionStatePairs=[]
        self.g=gameEnv()
        self.root=AnyNode(name="root",inheritedN=0)

    def takeAction(self,dim,val,stop,N):
        self.g.step(dim,val,stop,N)

    def updatePartition(self,dim,val,obs):
        self.ActionStatePairs.append([dim,val,obs])



    def takeTempPolicy(self):
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
            inheritedN=self.g.childNodeN(parentNode.inheritedN)
            nodes[i].name = str(i)
            nodes[i].dim = str(li[i][0])
            nodes[i].val = str(li[i][1])
            nodes[i].parent = parentNode
            nodes[i].inheritedN = inheritedN

            self.takeAction(li[i][0],li[i][1],li[i][3],inheritedN)

        self.g.renderEnv()
            #print(nameNode,"yo")
            #AnyNode(name="sub0B", parent=s0, index=1, partition=9)
        DotExporter(self.root).to_picture("TreeDiag/root.png")





if __name__== "__main__":
    a=agent()
    a.takeTempPolicy()
