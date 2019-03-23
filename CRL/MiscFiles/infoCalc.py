# This function basically boosts the amount of "Ether" samples
# in the off chance that they got diluted after
# successive cuts
def _boostingInheritedN(df,inheritedN):
    Y=len(df.index)
    # We have handled the case in the previous env class,
    # where the self.df has zero elements, thus Y>0
    if(inheritedN<=Y):
        # This can be defined to any real number
        c=1
        inheritedN=c*Y

    return inheritedN


def _info(df,inheritedN):
    #print("inherited N", inheritedN)
    Y=len(df.index)
    if(Y==0):
        ExpectedInfo = 0
        return ExpectedInfo

    probOfY=Y/(inheritedN+Y)
    probOfN=1-probOfY

    #print("printing probOfY,inheritedN,Y, probOfN",probOfY,inheritedN,Y,probOfN)
    infoY = -1*math.log(probOfY)
    infoN = -1*math.log(probOfN)
    ExpectedInfo = probOfY*infoY + probOfN*infoN
    # Note: information is always positive
    return ExpectedInfo

def _infoAfterPartition(ModD1,D1Info,ModD2,D2Info):

    PartInfo = ( ( ModD1*D1Info + ModD2*D2Info )/ (ModD1+ModD2) )
    return PartInfo

def _infoGain(self,df,dim,val,start,end,inheritedN):

    # we split our space based on the action taken
    df1=df.loc[df[str(dim)] >= val]
    df2=df.loc[df[str(dim)] < val]

    ModD1=len(df1.index)
    ModD2=len(df2.index)


    # This is based on the cluster Tree approach to update
    # the inheritedN
    inheritedN=self._boostingInheritedN(df,inheritedN)
    N1=((end-val)*inheritedN)/(end-start)
    N2=((val-start)*inheritedN)/(end-start)

    D1Info=_info(df1,N1)
    D2Info=_info(df2,N2)
    origInfo = _info(df,inheritedN)
    PartInfo=_infoAfterPartition(ModD1,D1Info,ModD2,D2Info)


    InfoGain = origInfo - PartInfo
    # Note: information is always positive, but delta info
    return InfoGain,df1,df2,N1,N2
