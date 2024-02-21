import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def discriminant_value(logdetOfcov,cov_inf,cov,mean,test_sample,prior):
    return (-0.5*(logdetOfcov) - 0.5*( np.dot(np.dot((test_sample).T,cov_inf),(test_sample)) - 2*np.dot(np.dot(np.transpose(mean),cov_inf),(test_sample)) + np.dot(np.dot(np.transpose(mean),cov_inf),mean)) + np.log(prior))

def QDA(x_train_reshaped, y_train, x_test_reshaped, y_test):
    x_train_reshaped = x_train_reshaped.astype(np.float64)/784
    x_test_reshaped = x_test_reshaped.astype(np.float64)/784

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
    for i in range(len(x_test_reshaped)):
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
    print("Total Accuracy: ",correct/sum(total_classified)*100,"%")
    # for i in range(10):
    #     print("Accuracy for class ",i,": ",correctly_classified[i]/total_classified[i]*100)


with np.load('mnist.npz') as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test,y_test = f["x_test"], f["y_test"]
x_train_reshaped = x_train.reshape(x_train.shape[0], -1).astype(np.float64)/255
x_test_reshaped = x_test.reshape(x_test.shape[0], -1).astype(np.float64)/255

mat=[]
new_y_train=[]
for i in range(10):
    count=0
    for j in range(100):
        while(y_train[count]!=i):
            count+=1
        mat.append(x_train_reshaped[count])
        new_y_train.append(i)
        count+=1
X=np.array(mat)
X=X.T
                       
mean=np.mean(X,axis=0)
X=X-np.mean(X,axis=0)
S=np.dot(X,X.T)/999
eig_val, eig_vec = np.linalg.eigh(S)
idx = eig_val.argsort()[::-1]
eig_val = eig_val[idx]
eig_vec = eig_vec[:,idx]
U=eig_vec

# X=X+mean
Y=np.dot(U.T,X)
X_recon=np.dot(U,Y)
MSE=0
for i in range(784):
    for j in range(1000):
        MSE+=np.sum((X[i][j]-X_recon[i][j])**2)   
print(MSE)

ps=[5,10,20,400,784]
for p in ps:
    print(np.dot(U[:,0:p],Y[:p]).shape)
    Y=np.dot(U.T,X)
    fig, ax = plt.subplots(10, 5, figsize=(10, 10))
    fig.suptitle(f'5 Samples of each class for p: {p}')
    plt.tight_layout()

    for i in range(10):
        ax[i, 0].set_title(f'class: {i}')
        for j in range(5):
            ax[i,j].imshow(np.reshape(np.dot(U[:,0:p],Y[:p])[:,j+(i*100)],(28,28)),cmap='gray',extent=None)
            ax[i][j].axis('off')
    # plt.get_current_fig_manager().window.state('zoomed')
    plt.show()

new_y_train=np.array(new_y_train)

for p in ps:
    print("for p: ",p)
    reduced_x_test = np.dot(U[:, :p].T, (x_test_reshaped.T-np.mean(x_test_reshaped.T,axis=0)))
    reduced_x_train=np.dot(U[:, :p].T,X)
    # print(Y_test.shape)
    QDA(reduced_x_train.T,new_y_train,reduced_x_test.T,y_test)
