import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with np.load('mnist.npz') as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
x_train = x_train.reshape(x_train.shape[0],-1).astype(np.float64)/255
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float64)/255

x_train_recon=[]
y_test_recon=[]
x_test_recon=[]
y_train_recon=[]
x_val_recon=[]
y_val_recon=[]
weights=[]

count1=0
count0=0
for i in range(60000):
    if y_train[i]==0:
        if(count0<1000):
            x_val_recon.append(x_train[i])
            y_val_recon.append(-1)
            count0+=1
        else:
            x_train_recon.append(x_train[i])
            y_train_recon.append(-1)
    elif y_train[i]==1:
        if(count1<1000):
            x_val_recon.append(x_train[i])
            y_val_recon.append(1)
            count1+=1
        else:
            x_train_recon.append(x_train[i])
            y_train_recon.append(1)

for i in range(10000):
    if y_test[i]==0:
        x_test_recon.append(x_test[i])
        y_test_recon.append(-1)
    if y_test[i]==1:
        x_test_recon.append(x_test[i])
        y_test_recon.append(y_test[i])

x_train=np.array(x_train_recon)
y_train=np.array(y_train_recon)
x_test=np.array(x_test_recon)
y_test=np.array(y_test_recon)
x_val=np.array(x_val_recon)
y_val=np.array(y_val_recon)

print(f'X_train_reshaped shape: {x_train.shape}')
X=x_train
X=X.T   

mean=np.mean(X,axis=1 , keepdims=True)  
X=X-mean   
S=np.dot(X,X.T)/(X.shape[1]-1)  
eig_val,eig_vec=np.linalg.eig(S)    
idx=np.argsort(eig_val)[::-1]
eig_val=eig_val[idx]
eig_vec=eig_vec[:,idx]
U=eig_vec     
p=5
Up = U[:, :p]  
Y = np.dot(Up.T,X)
x_val = np.dot(U[:,  :p].T, (x_val.T-mean))
x_test = np.dot(U[:,  :p].T, (x_test.T-mean))
print(Y.shape)
print(x_val.shape)

class Node:
    def __init__(self,feature_index=None,threshold=None,left=None,right=None,value=None,weight=None):
        self.feature_index=feature_index
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value
        self.weight=weight

def decision_stump(stumpNo,Y,y_train,x_val,y_val,MSEs,h,h1):
    mn=1000000000
    for i in range(5):
        unique_values=[]
        for j in range(y_train.size):
            unique_values.append(Y[i][j])
        unique_values=np.array(unique_values)
        unique_values.sort(axis=0)
        case_indexes=[]
        for j in range(1000):
            case_indexes.append(random.randint(0,y_train.size-2))
        for j in case_indexes:
            split=(unique_values[j]+unique_values[j+1])/2
            left_sum=0
            left_total=0
            right_sum=0
            right_total=0
            for k in range(y_train.size):
                if(Y[i][k]<=split):
                    left_sum+=y_train[k]
                    left_total+=1
                else:
                    right_sum+=y_train[k]
                    right_total+=1
                    
            left_mean=left_sum/left_total
            right_mean=right_sum/right_total
            # print(left_sum,left_total)
            # print(right_sum,right_total)
            # print(mn)
                
            SSR=0
            for k in range(y_train.size):
                if(Y[i][k]<=split):
                    SSR+=(y_train[k]-left_mean)**2
                else:
                    SSR+=(y_train[k]-right_mean)**2

            # print(SSR)
            if(SSR<mn):
                mn=SSR
                ans=[i,split,left_mean,right_mean]

    # print(ans[2],ans[3])
    temp=[]
    for i in range(y_train.size):
        if(Y[ans[0]][i]<=ans[1]):
            temp.append(ans[2])
        else:
            temp.append(ans[3])
    
    h.append(temp)


    residues=[]
    for i in range(y_train.size):
        residues.append(y_train[i])
    
    for i in range(stumpNo+1):
        for j in range(y_train.size):
            residues[j]-=(0.01)*h[i][j]
    
    for i in range(y_train.size):
        y_train[i]=np.sign(residues[i])


    temp=[]
    for i in range(y_val.size):
        if(x_val[ans[0]][i]<=ans[1]):
            temp.append(ans[2])
        else:
            temp.append(ans[3])
    h1.append(temp)
                                          
    MSE=0

    for j in range(y_val.size):
        temp=y_val[j]
        for i in range(stumpNo+1):
            temp-=(0.01)*h1[i][j]
        MSE+=temp**2

    # for i in range(y_val.size):
    #     MSE+=(y_val[i]-(0.01)*temp[i])**2

    MSE/=y_val.size
    print(MSE)

    return ans





h=[]
h1=[]
MSEs=[]
stumps=[]

noOfStumps=300
for i in range(noOfStumps):
    stumps.append(decision_stump(i,Y,y_train=y_train,x_val=x_val,y_val=y_val,MSEs=MSEs,h=h,h1=h1))

print(MSEs)

# minMSE=0
# for i in range(noOfStumps):
#     if(minMSE<MSEs[i]):
#         minMSE=MSEs[i]
#         best_stump=stumps[i]

# # best_stump=stumps[accuracy_values.index(max(accuracy_values))]
# # print(best_stump)
# correct=0
# for i in range(y_test.size):
#     if(x_test[best_stump[0]][i]<=best_stump[1] and y_test[i]==best_stump[2]):
#         correct+=100
#     elif(x_test[best_stump[0]][i]>best_stump[1] and y_test[i]==best_stump[3]):
#         correct+=100
# print(f"accurracy on test set",correct/y_test.size)

stumps_range = range(1, noOfStumps+1)
plt.figure(figsize=(10, 5))
plt.plot(stumps_range, MSEs, marker='o')
plt.title('Validation Set Accuracy vs Number of Stumps')
plt.xlabel('Number of Stumps')
plt.ylabel('Accuracy (%)')
plt.xticks(stumps_range)
plt.ylim(95, 100)
plt.grid(True)
plt.show()
