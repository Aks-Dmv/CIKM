import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
N = 200
g1 = 2+np.random.rand(N)
g2 = -1+np.random.rand(N)
g3 = 3+np.random.rand(N)
g4 = 3 + np.random.rand(N)
g5 = 5 + np.random.rand(N)
g6 = 1 + np.random.rand(N)
x=np.append(g1,g2)
x=np.append(x,g3)

y=np.append(g4,g5)
y=np.append(y,g6)
data=np.array([x,y])
data=data.T
np.savetxt("dataPts.csv", data, delimiter=",")
print(data.shape)

df=pd.DataFrame(data)
#print(df)
#import matplotlib.pyplot as plt
plt.scatter(df[0], df[1])
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
