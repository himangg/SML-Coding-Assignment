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

# print(f'X_train_reshaped shape: {x_train.shape}')
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
# print(Y.shape)
# print(x_val.shape)

#assign initial weights
for i in range(Y[0].size):
    weights.append(1/y_train.size)

weights=np.array(weights)

class Node:
    def __init__(self,feature_index=None,threshold=None,left=None,right=None,value=None,weight=None):
        self.feature_index=feature_index
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value
        self.weight=weight


def decision_stump(stumpNo,Y,y_train,weights,x_val,individual_predictions,prev_sum,accuracy_values,alphas):
    mn=1
    for i in range(5):
        unique_values=[]
        for j in range(y_train.size):
            unique_values.append([Y[i][j],y_train[j]])
        unique_values=np.array(unique_values)
        unique_values=unique_values[np.argsort(unique_values[:, 0])]

        case_indexes=[]
        for j in range(1000):
            case_indexes.append(random.randint(0,y_train.size-2))

        for j in case_indexes:
            split=(unique_values[j][0]+unique_values[j+1][0])/2

            left_1 = np.sum(weights[(Y[i] <= split) & (y_train == 1)])
            left_0 = np.sum(weights[(Y[i] <= split) & (y_train == -1)])
            right_1 = np.sum(weights[(Y[i] > split) & (y_train == 1)])
            right_0 = np.sum(weights[(Y[i] > split) & (y_train == -1)])
                
            if(left_1>left_0):
                left_split_region_mode=1
            else:
                left_split_region_mode=-1
            if(right_1>right_0):
                right_split_region_mode=1
            else:
                right_split_region_mode=-1

            incorrect_mask = ((Y[i] <= split) & (y_train != left_split_region_mode)) | ((Y[i] > split) & (y_train != right_split_region_mode))
            incorrect_weights = weights[incorrect_mask]

            incorrect = np.sum(incorrect_weights)
            total = np.sum(weights)
            incorrectly_classified = np.where(incorrect_mask)[0]

            weighted_miss_classification_error=incorrect/total
            if(weighted_miss_classification_error<mn):
                mn=weighted_miss_classification_error
                ans_incorrectly_classified=incorrectly_classified
                ans=[i,split,left_split_region_mode,right_split_region_mode]

    alpha=math.log((1-mn)/mn)
    alphas.append(alpha)
    print(f"Alpha for tree",stumpNo+1,":",alpha)
    for i in ans_incorrectly_classified:
        weights[i]*=(1-mn)/mn

    for i in range(2000):
        if(x_val[ans[0]][i]<=ans[1]):
            individual_predictions[stumpNo][i]=alpha*ans[2]
        else:
            individual_predictions[stumpNo][i]=alpha*ans[3]
    
    correct=0
    for j in range(2000):
        prev_sum[j]+=individual_predictions[stumpNo][j]
        if(prev_sum[j]<0 and y_val[j]==-1):
            correct+=1
        elif(prev_sum[j]>0 and y_val[j]==1):
            correct+=1
    accuracy_values.append(correct/20)
    print(f"Accurracy for tree",stumpNo+1,":",correct/20)
    return ans


individual_predictions = np.array([[0.0 for i in range(2000)] for j in range(300)])

prev_sum=np.array([0.0 for i in range(2000)])
accuracy_values=[]
stumps=[]

noOfStumps=300
alphas=[]
for i in range(noOfStumps):
    stumps.append(decision_stump(i,Y,y_train,weights=weights,x_val=x_val,individual_predictions=individual_predictions,prev_sum=prev_sum,accuracy_values=accuracy_values,alphas=alphas))

# print(accuracy_values)

max_accuracy=0
for i in range(noOfStumps):
    if(max_accuracy<accuracy_values[i]):
        max_accuracy=accuracy_values[i] 
        best_stump=stumps[i]
        best_stump_no=i

# best_stump=stumps[accuracy_values.index(max(accuracy_values))]
# print(best_stump)

individual_predictions_test=np.array([[0.0 for i in range(y_test.size)] for j in range(300)])
prev_sum_test=np.array([0.0 for i in range(y_test.size)])
for k in range(best_stump_no+1):
    for i in range(y_test.size):
        if(x_test[stumps[k][0]][i]<=stumps[k][1]):
            individual_predictions_test[k][i]=alphas[k]*stumps[k][2]
        else:
            individual_predictions_test[k][i]=alphas[k]*stumps[k][3]
    for j in range(y_test.size):
        prev_sum_test[j]+=individual_predictions_test[k][j]

correct=0
for j in range(y_test.size):
    if(prev_sum_test[j]<0 and y_test[j]==-1):
        correct+=1
    elif(prev_sum_test[j]>0 and y_test[j]==1):
        correct+=1
print(f"Accurracy on Test Set:",correct/y_test.size)

stumps_range = range(1, noOfStumps+1)
plt.figure(figsize=(10, 5))
plt.plot(stumps_range, accuracy_values, marker='o')
plt.title('Validation Set Accuracy vs Number of Stumps')
plt.xlabel('Number of Stumps')
plt.ylabel('Accuracy (%)')
plt.xticks(range(0, noOfStumps+1, 25))
plt.ylim(99, 100)
plt.grid(True)
plt.show()
