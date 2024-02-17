import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def discriminant_value(cov_inf,covariance,mean,test_sample,prior):
    return (-0.5*(np.linalg.slogdet(covariance))[1] - 0.5*( np.dot(np.dot((test_sample).T,cov_inf),(test_sample)) - 2*np.dot(np.dot(np.transpose(mean),cov_inf),(test_sample)) + np.dot(np.dot(np.transpose(mean),cov_inf),mean)) + np.log(prior))
mnist_data = np.load("C:/Users/himan/Desktop/SML_ASSIGNMENT_2/mnist.npz")
x_train = mnist_data['x_train']
y_train = mnist_data['y_train']
x_test = mnist_data['x_test']
y_test = mnist_data['y_test']

for i in range(10):
    fig=plt.figure(figsize=(12,4))
    fig.suptitle('5 Samples of '+str(i))
    count=0
    for j in range(5):
        while(y_train[count]!=i):
            count+=1
        fig.add_subplot(1, 5, j+1)
        plt.imshow(x_train[count],cmap='Grays')
        count+=1
    plt.show()

x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
training_classes=[]
for i in range(10):
    training_classes.append(x_train_reshaped[np.where(y_train == i)[0]])
priors=[]
for i in range(10):
    priors.append(len(x_train_reshaped[y_train==i])/len(x_train_reshaped))
covariances = []
for i in range(10):
    covariance=np.cov(training_classes[i].T)
    if(np.linalg.det(covariance)==0):
        lambdaa=0.000001
        covariance+=lambdaa*np.identity(covariance.shape[0])
    covariances.append(covariance)
covariances_inverse = []
for i in range(10):
    covariances_inverse.append(np.linalg.inv(covariances[i]))
means=[]
for i in range(10):
    means.append(np.mean(training_classes[i],axis=0))
correct=0
for i in range(10000):
    discriminant_values = []
    max_value=discriminant_value(covariances_inverse[0],covariances[0],means[0],x_test_reshaped[i],priors[0])
    ans=0
    for j in range(1,10):
        if(discriminant_value(covariances_inverse[j],covariances[j],means[j],x_test_reshaped[i],priors[j])>max_value):
            max_value=discriminant_value(covariances_inverse[j],covariances[j],means[j],x_test_reshaped[i],priors[j])
            ans=j
    if(ans==y_test[i]):
        correct+=1
    else:
        print("Misclassified: ",i)
print("Accuracy: ",correct)
