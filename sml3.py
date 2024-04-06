import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

with np.load('mnist.npz') as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
x_train_reshaped = x_train.reshape(x_train.shape[0],-1).astype(np.float64)/255
x_test_reshaped = x_test.reshape(x_test.shape[0], -1).astype(np.float64)/255

x_train_reshaped_recon=[]
y_test_recon=[]
x_test_reshaped_recon=[]
y_train_recon=[]

for i in range(60000):
    if y_train[i]==0 or y_train[i]==1 or y_train[i]==2:
        x_train_reshaped_recon.append(x_train_reshaped[i])
        y_train_recon.append(y_train[i])

for i in range(10000):
    if y_test[i]==0 or y_test[i]==1 or y_test[i]==2:
        x_test_reshaped_recon.append(x_test_reshaped[i])
        y_test_recon.append(y_test[i])

x_train_reshaped=np.array(x_train_reshaped_recon)
y_train=np.array(y_train_recon)
x_test_reshaped=np.array(x_test_reshaped_recon)
y_test=np.array(y_test_recon)

print(f'X_train_reshaped shape: {x_train_reshaped.shape}')
X=x_train_reshaped
X=X.T

mean=np.mean(X,axis=1 , keepdims=True)  
X=X-mean   
S=np.dot(X,X.T)/(X.shape[1]-1)  
eig_val,eig_vec=np.linalg.eig(S)    
idx=np.argsort(eig_val)[::-1]
eig_val=eig_val[idx]
eig_vec=eig_vec[:,idx]
U=eig_vec     
p=10
Up = U[:, :p]  
Y = np.dot(U[:, :p].T,X)
x_test_reshaped_pca = np.dot(U[:,  :p].T, (x_test_reshaped.T-mean)).T
print(x_test_reshaped_pca.shape)


class Node:
    def __init__(self,feature_index=None,threshold=None,left=None,right=None,value=None):
        self.feature_index=feature_index
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

def split_region(Y,feature_index,threshold):
    x_left_split_region=np.array([])
    x_right_split_region=np.array([])
    y_left_split_region=np.array([])
    y_right_split_region=np.array([])
    for i in range(Y[0].size):
        if(Y[feature_index][i]<=threshold):
            left_split_region.append(Y.T[i])
        else:
            right_split_region.append(Y.T[i])
    left_split_region=np.array(left_split_region)
    right_split_region=np.array(right_split_region)
    return left_split_region,right_split_region

def build_tree(Y,y_train):
    min_gini=float('inf')
    for pi in range(10):
        pmk=[0,0,0]
        for i in range(Y[0].size):
            if(Y[pi][i]<=np.mean(Y[pi])):
                pmk[y_train[i]]+=1
        pmk1=[0,0,0]
        pmk1=pmk/np.sum(pmk)
        gini1=pmk1[0]*(1-pmk1[0])+pmk1[1]*(1-pmk1[1])+pmk1[2]*(1-pmk1[2])
        pmk=[0,0,0]
        for i in range(Y[0].size):
            if(Y[pi][i]>np.mean(Y[pi])):
                pmk[y_train[i]]+=1
        pmk2=[0,0,0]
        pmk2=pmk/np.sum(pmk)
        gini2=pmk2[0]*(1-pmk2[0])+pmk2[1]*(1-pmk2[1])+pmk2[2]*(1-pmk2[2])
        gini=min(gini1,gini2)
        if(gini<min_gini):
            min_gini=gini
            feature_index=pi
            threshold=np.mean(Y[pi])
        print(gini)
    decision_tree=Node(feature_index=feature_index,threshold=threshold)
    return decision_tree

def build_terminal_node(node1,Y,y_train):
    x_left_split_region=[]
    x_right_split_region=[]
    y_left_split_region=[]
    y_right_split_region=[]

    for i in range(Y[0].size):
        if(Y[node1.feature_index][i]<=node1.threshold):
            x_left_split_region.append(Y.T[i])
            y_left_split_region.append(y_train[i])
        else:
            x_right_split_region.append(Y.T[i])
            y_right_split_region.append(y_train[i])

    x_left_split_region=np.array(x_left_split_region)
    y_left_split_region=np.array(y_left_split_region)
    x_right_split_region=np.array(x_right_split_region)
    y_right_split_region=np.array(y_right_split_region)
    terminal_node_left=Node(value=stats.mode(y_left_split_region))
    terminal_node_right=Node(value=stats.mode(y_right_split_region))
    return [terminal_node_left,terminal_node_right]

def build_final_tree(Y,y_train):
    dec_node1 = build_tree(Y,y_train)    
    x_left_split_region=[]
    x_right_split_region=[]
    y_left_split_region=[]
    y_right_split_region=[]

    for i in range(Y[0].size):
        if(Y[dec_node1.feature_index][i]<=dec_node1.threshold):
            x_left_split_region.append(Y.T[i])
            y_left_split_region.append(y_train[i])
        else:
            x_right_split_region.append(Y.T[i])
            y_right_split_region.append(y_train[i])

    x_left_split_region=np.array(x_left_split_region)
    y_left_split_region=np.array(y_left_split_region)
    x_right_split_region=np.array(x_right_split_region)
    y_right_split_region=np.array(y_right_split_region)

    dec_node2=build_tree(x_left_split_region.T,y_left_split_region)
    dec_node1.left=dec_node2
    dec_node1.right=build_terminal_node(dec_node1,Y,y_train)[1]
    dec_node2.left=build_terminal_node(dec_node2,x_left_split_region.T,y_left_split_region)[0]
    dec_node2.right=build_terminal_node(dec_node2,x_left_split_region.T,y_left_split_region)[1]
    return dec_node1

dec_node1=build_final_tree(Y,y_train)
# print(dec_node1.feature_index)
# print(dec_node1.threshold)
# print(dec_node1.left.feature_index)
# print(dec_node1.left.threshold)
# print(dec_node1.right.value)
# print(dec_node1.left.left.value)
# print(dec_node1.left.right.value)

# 2nd part
class_counts = {0: {'correct': 0, 'total': 0}, 1: {'correct': 0, 'total': 0}, 2: {'correct': 0, 'total': 0}}
total_correct = 0
total_instances = y_test.size
for i in range(y_test.size):
    class_counts[y_test[i]]['total'] += 1
    if(x_test_reshaped_pca[i][dec_node1.feature_index] >= dec_node1.threshold):
        if(dec_node1.right.value[0] == y_test[i]):
            class_counts[y_test[i]]['correct'] += 1
            total_correct += 1
    else:
        if(x_test_reshaped_pca[i][dec_node1.left.feature_index] >= dec_node1.left.threshold):
            if(dec_node1.left.right.value[0] == y_test[i]):
                class_counts[y_test[i]]['correct'] += 1
                total_correct += 1
        else:
            if(dec_node1.left.left.value[0] == y_test[i]):
                class_counts[y_test[i]]['correct'] += 1
                total_correct += 1
for class_label, counts in class_counts.items():
    accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
    print(f"Accuracy for class {class_label}: {accuracy:.2%}")
total_accuracy = total_correct / total_instances if total_instances > 0 else 0
print(f"Total Accuracy: {total_accuracy:.2%}")

# 3rd part

def resample(X, y):
    np.random.seed()
    indices = np.random.choice(len(X) ,len(X), replace=True)
    return X[indices], y[indices]

# indices = np.random.randint(0, len(Y), len(Y))

new_datasets=[]
for _ in range(5):
    X_boot, y_boot = resample(x_train_reshaped, y_train)
    X_boot = np.dot(U[:, :p].T,X_boot.T-mean).T
    new_datasets.append((X_boot, y_boot))

print(new_datasets[1][0].shape)
    
decision_trees = []
for X_boot, y_boot in new_datasets:
    dec_node = build_final_tree(X_boot.T, y_boot)
    decision_trees.append(dec_node)

def majority_vote(predictions):
    counts = Counter(predictions)
    return counts.most_common(1)[0][0]

def predict(test_sample):
    predictions = []
    for tree in decision_trees:
        if test_sample[tree.feature_index] >= tree.threshold:
            predictions.append(tree.right.value[0])
        else:
            if test_sample[tree.left.feature_index] >= tree.left.threshold:
                predictions.append(tree.left.right.value[0])
            else:
                predictions.append(tree.left.left.value[0])
    return majority_vote(predictions)

def accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_pred)
    return correct / total

test_predictions = [predict(sample) for sample in x_test_reshaped_pca]
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy:", accuracy)

class_accuracies = {}
for class_label in np.unique(y_test):
    indices = np.where(y_test == class_label)
    correct_predictions = np.sum(y_test[indices] == np.array(test_predictions)[indices])
    total_samples = len(indices[0])
    class_accuracy = correct_predictions / total_samples
    class_accuracies[class_label] = class_accuracy

print("\nClass-wise accuracies:")
for class_label, accuracy in class_accuracies.items():
    print(f"Class {class_label}: {accuracy}")
