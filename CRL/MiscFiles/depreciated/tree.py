from anytree import AnyNode, RenderTree
from anytree.exporter import DotExporter
import graphviz
# We need the attributes to have
# a max & a min value for each index
# Note: we can't have the partition as a
# variable because there may not be one partition per
# node
root = AnyNode(name="Akshay")
s0 = AnyNode(name="sub0", parent=root)
s0b = AnyNode(name="sub0B", parent=s0, index=1, partition=9)
s0a = AnyNode(name="sub0A", parent=s0)
s1 = AnyNode(name="sub1", parent=root)
s1a = AnyNode(name="sub1A", parent=s1)
s1b = AnyNode(name="sub1B", parent=s1, index=1)
s1c = AnyNode(name="sub1C", parent=s1)
s1ca = AnyNode(name="sub1Ca", parent=s1c)

#print(RenderTree(root))
s0.name="yoNig"
for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, node.name))


DotExporter(root).to_picture("root.png")


def infoGain(self,df,dim,val,start,end,inheritedN):
    # class importance is an array that
    # gives us the linear combination coeff
    # This can be given during runtime
    df1=df.loc[df[str(dim)] > val]
    df2=df.loc[df[str(dim)] < val]
    ModD1=len(df1.dim)
    ModD2=len(df2.dim)
    ModD=len(df.dim)
    N1=((end-val)*inheritedN)/(end-start)
    N2=((val-start)*inheritedN)/(end-start)

    deltaInfo = -1*( ModD1*info(df1,N1)+ModD2*info(df2,N2) )/ModD
    return deltaInfo
