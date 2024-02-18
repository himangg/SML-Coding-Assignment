import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
with np.load('mnist.npz') as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test,y_test = f["x_test"], f["y_test"]
x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
mat=[]
for i in range(10):
    count=0
    for j in range(100):
        while(y_train[count]!=i):
            count+=1
        mat.append(x_train_reshaped[count])
        count+=1
X=np.array(mat)
X=X.T
# print(X.shape)
# print(X.mean())

mean=np.mean(X,axis=0)
X=X-np.mean(X,axis=0)
# print(sum(sum(X)))
print(X)
S=np.dot(X,X.T)/999
print(S)
eig_val, eig_vec = np.linalg.eig(S)
# print(eig_val)
# print(eig_vec)
idx = eig_val.argsort()[::-1]
eig_val = eig_val[idx]
eig_vec = eig_vec[:,idx]
U=eig_vec

print(U)
Y=np.dot(U.T,X)
X_recon=np.dot(U,Y)
print(Y)
MSE=0
for i in range(784):
    for j in range(1000):
        MSE+=np.sum((X[i][j]-X_recon[i][j])**2)   
print(MSE)
# for p in range(5):

# u=U[:5]
# np.dot(u,Y)
# fig=plt.figure(figsize=(12,4))
# fig.add_subplot(1, 5, j+1)
# plt.imshow(np.dot(u,Y)[0].reshape(),cmap='Grays')
# plt.show()
# X+=mean
