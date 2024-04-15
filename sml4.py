import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
import math

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

count=0
for i in range(60000):
    if(count<1000):
        if y_train[i]==0:
            x_val_recon.append(x_train[i])
            y_val_recon.append(-1)
            count+=1
        if y_train[i]==1:
            x_val_recon.append(x_train[i])
            y_val_recon.append(y_train[i])
            count+=1
    else:
        if y_train[i]==0:
            x_train_recon.append(x_train[i])
            y_train_recon.append(-1)
        if y_train[i]==1:
            x_train_recon.append(x_train[i])
            y_train_recon.append(y_train[i])

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

#assign initial weights
for i in range(Y[0].size):
    weights.append(1/y_train.size)

class Node:
    def __init__(self,feature_index=None,threshold=None,left=None,right=None,value=None,weight=None):
        self.feature_index=feature_index
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value
        self.weight=weight


def decision_stump(stumpNo,Y,y_train,weights,x_val,individual_predictions,prev_sum,accuracy_values):
    mn=1
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
            left_1=0
            left_0=0
            right_1=0
            right_0=0
            for k in range(y_train.size):
                if(Y[i][k]<=split):
                    if(y_train[k]==1):
                        left_1+=weights[k]
                    else:
                        left_0+=weights[k]
                else:
                    if(y_train[k]==1):
                        right_1+=weights[k]
                    else:
                        right_0+=weights[k]
                
            if(left_1>left_0):
                left_split_region_mode=1
            else:
                left_split_region_mode=-1
            if(right_1>right_0):
                right_split_region_mode=1
            else:
                right_split_region_mode=-1

            incorrectly_classified=[]
            incorrect=0
            total=0
            for k in range(y_train.size):
                total+=weights[k]
                if(Y[i][k]<=split and y_train[k]!=left_split_region_mode):
                    # print(Y[i][k])    
                    incorrect+=weights[k]
                    incorrectly_classified.append(k)
                elif(Y[i][k]>split and y_train[k]!=right_split_region_mode):
                    # print(Y[i][k])
                    incorrect+=weights[k]
                    incorrectly_classified.append(k)
            weighted_miss_classification_error=incorrect/total
            if(weighted_miss_classification_error<mn):
                mn=weighted_miss_classification_error
                ans_incorrectly_classified=incorrectly_classified
                ans=[i,split,left_split_region_mode,right_split_region_mode]

    alpha=math.log((1-mn)/mn)
    print(f"alpha:",alpha)
    for i in ans_incorrectly_classified:
        weights[i]*=(1-mn)/mn

    for i in range(1000):
        if(x_val[ans[0]][i]<=ans[1]):
            individual_predictions[stumpNo][i]=alpha*ans[2]
        else:
            individual_predictions[stumpNo][i]=alpha*ans[3]
    
    correct=0
    for j in range(1000):
        prev_sum[j]+=individual_predictions[stumpNo][j]
        if(prev_sum[j]<0 and y_val[j]==-1):
            correct+=1
        elif(prev_sum[j]>0 and y_val[j]==1):
            correct+=1
    accuracy_values.append(correct/10)
    print(f"accurracy",correct/10)
    return ans


individual_predictions = np.array([[0.0 for i in range(1000)] for j in range(300)])

prev_sum=np.array([0.0 for i in range(1000)])
accuracy_values=[]
stumps=[]

noOfStumps=300
for i in range(noOfStumps):
    stumps.append(decision_stump(i,Y,y_train,weights=weights,x_val=x_val,individual_predictions=individual_predictions,prev_sum=prev_sum,accuracy_values=accuracy_values))

# print(accuracy_values)

max_accuracy=0
for i in range(noOfStumps):
    if(max_accuracy<accuracy_values[i]):
        max_accuracy=accuracy_values[i]
        best_stump=stumps[i]

# best_stump=stumps[accuracy_values.index(max(accuracy_values))]
# print(best_stump)
correct=0
for i in range(y_test.size):
    if(x_test[best_stump[0]][i]<=best_stump[1] and y_test[i]==best_stump[2]):
        correct+=100
    elif(x_test[best_stump[0]][i]>best_stump[1] and y_test[i]==best_stump[3]):
        correct+=100
print(f"accurracy on test set",correct/y_test.size)

stumps_range = range(1, noOfStumps+1)
plt.figure(figsize=(10, 5))
plt.plot(stumps_range, accuracy_values, marker='o')
plt.title('Validation Set Accuracy vs Number of Stumps')
plt.xlabel('Number of Stumps')
plt.ylabel('Accuracy (%)')
plt.xticks(stumps_range)
plt.ylim(95, 100)
plt.grid(True)
plt.show()
