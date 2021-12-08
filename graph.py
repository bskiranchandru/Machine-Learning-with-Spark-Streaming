import matplotlib.pyplot as plt
import numpy as np
  
  
# Define X and Y variable data
file1 = open("acc_store.txt", "r") 
file2 = open("acc_store1.txt", "r")
file3 = open("acc_store2.txt", "r")
y=[]
z=[]
a=[]
for i in file1:
    y.append(float(i))
for i in file2:
    z.append(float(i))
for i in file3:
    a.append(float(i))
 
x =list(range(len(y)))
plt.plot(x,y,label='MNB')
plt.plot(x,z,label='Passiveaagressive')
plt.plot(x,a,label='perceptron')
plt.xlabel("batch")  # add X-axis label
plt.ylabel("accuracy")  # add Y-axis label
plt.title("analytics")  # add title
plt.legend()
plt.show()