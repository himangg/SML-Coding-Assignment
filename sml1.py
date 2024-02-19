import numpy as np
import matplotlib.pyplot as plt
def discriminant_value(logdetOfcov,cov_inf,cov,mean,test_sample,prior):
    return (-0.5*(logdetOfcov) - 0.5*( np.dot(np.dot((test_sample).T,cov_inf),(test_sample)) - 2*np.dot(np.dot(np.transpose(mean),cov_inf),(test_sample)) + np.dot(np.dot(np.transpose(mean),cov_inf),mean)) + np.log(prior))

mnist_data = np.load("C:/Users/himan/Desktop/SML_ASSIGNMENT_2/mnist.npz")
x_train = mnist_data['x_train'].astype(np.float64)/255
y_train = mnist_data['y_train'].astype(np.float64)
x_test = mnist_data['x_test'].astype(np.float64)/255
y_test = mnist_data['y_test'].astype(np.float64)

fig, axes = plt.subplots(10, 5, figsize=(12, 20))
fig.suptitle('5 Samples of each class')
for i in range(10):
    axes[i, 0].set_title(f'Class {i}')
    count = 0
    plt.tight_layout(pad=3.0)
    for j in range(5):
        while y_train[count] != i:
            count += 1
        axes[i, j].imshow(x_train[count], cmap='gray')
        axes[i, j].axis('off')
        count += 1
# plt.get_current_fig_manager().window.state('zoomed')
plt.show()

x_train_reshaped = x_train.reshape(x_train.shape[0], -1).astype(np.float64)
x_test_reshaped = x_test.reshape(x_test.shape[0], -1).astype(np.float64)

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

logdetOfCovariances = []
for i in range(10):
    logdetOfCovariances.append(np.linalg.slogdet(covariances[i])[1])

correct=0
correctly_classified=[0,0,0,0,0,0,0,0,0,0]
total_classified=[0,0,0,0,0,0,0,0,0,0]
for i in range(10000):
    discriminant_values = []
    max_value=discriminant_value(logdetOfCovariances[0],covariances_inverse[0],covariances[0],means[0],x_test_reshaped[i],priors[0])
    ans=0
    for j in range(1,10):
        if(discriminant_value(logdetOfCovariances[j],covariances_inverse[j],covariances[j],means[j],x_test_reshaped[i],priors[j])>max_value):
            max_value=discriminant_value(logdetOfCovariances[j],covariances_inverse[j],covariances[j],means[j],x_test_reshaped[i],priors[j])
            ans=j
    if(ans==y_test[i]):
        correctly_classified[int(ans)]+=1
        total_classified[int(ans)]+=1
        correct+=1
        # print("Correctly classified: ",i)
    else:
        # print("Misclassified: ",i)
        total_classified[int(y_test[i])]+=1
print("Total Accuracy: ",correct/sum(total_classified)*100)
for i in range(10):
    print("Accuracy for class ",i,": ",correctly_classified[i]/total_classified[i]*100,"%")
