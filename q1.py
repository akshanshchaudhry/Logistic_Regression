#Akshansh Chaudhry

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#funtion to the required data set
def generate_data(mean,cov,size):
    return (np.random.multivariate_normal(mean,cov,size))

#function for batch training. It returns the final weights and iterations
def neuron_batch_train(data,weights,iterations,target,learning_rate):
        for j in range (0, iterations):
            err = 0
            output_list=[]
            for i in range (0, len(data)):
                temp_netvalue = np.dot(data[i], weights[:2]) + weights[-1]
                output = sigmoid_function(temp_netvalue)
                output_list.append(output)
                err = err + cross_entropy(output,target[i])/len(data)
            gd = np.concatenate((gradiant_decent(np.reshape(output_list,[len(output_list),1]),data,np.reshape(target,[len(target),1])),np.reshape(weights[-1],[len(weights[-1]),1])),axis=0)/len(data)
            weights = weights_updation(old_weights=weights,gradiant_decent=gd,learning_rate=learning_rate)

            if(err[0]<0.1 or np.average(gd)<0.001):
                break
        return weights,(j+1)

#function for online training. It returns the final weights and iterations
def neuron_online_train(data,weights,iterations,target,learning_rate):

        for j in range (0, iterations):
            err = 0
            for i in range (0, len(data)):
                temp_netvalue = np.dot(data[i], weights[:2]) + weights[-1]
                output = sigmoid_function(temp_netvalue)
                err = cross_entropy(output,target[i])/len(data)
                gradiant_decent = (output-target[i])*data[i]
                gradiant_decent = np.reshape(gradiant_decent,[len(gradiant_decent),1])
                gradiant_decent = np.concatenate((gradiant_decent,np.reshape(weights[-1],[len(weights[-1]),1])),axis=0)
                weights = weights_updation(old_weights=weights,gradiant_decent=gradiant_decent,learning_rate=learning_rate)

            if(err[0]<0.00000000001):
                break
        return weights,(j+1)

#function for sigmoid function
def sigmoid_function(x):
    netvalue = 1/(1+(np.exp(1)**(-x)))
    return netvalue

#function to calculate cross entropy
def cross_entropy(output,target):
    return -((target* np.log(output))+((1-target)*np.log(1-output)))

#function to calculate gradiant decent
def gradiant_decent(output,input,target):
    return np.dot(np.transpose(input),np.subtract(output,target))

#funtion to update weights
def weights_updation(old_weights,gradiant_decent,learning_rate):
    return (old_weights - (learning_rate * gradiant_decent))

#function for prediction. It returns the accuracy and the list of predicted values
def prediction(test_data,weights,test_targets):
    output_list=[]
    true_positive_false_negative_count = 0
    true_positive_rate = 0
    for i in range(len(test_data)):
        netvalue = np.dot(test_data[i], weights[:2]) + weights[-1]
        output= sigmoid_function(netvalue)
        if output > 0.5:
            output = 1
            output_list.append(output)
        else:
            output = 0
            output_list.append(output)

        if output == test_targets[i]:
            true_positive_false_negative_count += 1

    c_m = confusion_matrix(test_targets,output_list)
    accuracy = ((c_m[0][0]+c_m[1][1])/(c_m[0][0]+c_m[1][0]+c_m[0][1]+c_m[1][1]))*100

    return accuracy,output_list

#function for confusion matrix
def confusion_matrix(actual,predicted):
    actual = pd.Series(actual, name='Actual')
    predicted = pd.Series(predicted, name='Predicted')
    df_confusion = pd.crosstab(actual, predicted)
    return df_confusion

# My function for plotting ROC curve and calculating area under the curve
def roc_curve(tpr,fpr,c_m,learning_rate):
    plt.title('Roc Curve')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    auc = ((c_m[0][0]+c_m[1][1])/(c_m[0][0]+c_m[1][0]+c_m[0][1]+c_m[1][1]))*100
    plt.plot(fpr, tpr, label = 'Area Under Curve for learnig rate {}: {}'.format(learning_rate,auc))
    plt.fill_between(fpr,0,tpr)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_data_1 = generate_data([1,0],[[1,0.75],[0.75,1]],1000)
    train_data_2 = generate_data([0,1.5],[[1,0.75],[0.75,1]],1000)
    test_data_1 = generate_data([1,0],[[1,0.75],[0.75,1]],500)
    test_data_2 = generate_data([0,1.5],[[1,0.75],[0.75,1]],500)
    target = [0 for i in range(1000)] + [1 for j in range(1000)]
    target_test = [0 for i in range(500)] + [1 for j in range(500)]
    train_data = np.concatenate((train_data_1,train_data_2), axis=0)
    test_data = np.concatenate((test_data_1,test_data_2), axis=0)
    weights=[1,1,1]
    learning_rate = [1,0.1,0.01]
    final_weights_batch = []
    tpr_batch_list =[0,0,1]
    fpr_batch_list=[0,0,1]

    # performing batch training
    for i in learning_rate:
        trained_weights=neuron_batch_train(train_data, np.reshape(np.array([weights]),(3,1)), 10000, target,i)
        print("The accuracy for batch training for learning rate {} is: {}".format(i,prediction(test_data,trained_weights[0],target_test)[0]))
        print("The iterations are: {}".format(trained_weights[1]))
        print("The final weights are: {}".format(trained_weights[0]))
        c_m_temp = confusion_matrix(target_test,prediction(test_data,trained_weights[0],target_test)[1])
        tpr_batch_list[1] = c_m_temp[0][0]/(c_m_temp[0][0]+c_m_temp[1][0])
        fpr_batch_list[1] = c_m_temp[1][0]/(c_m_temp[1][0]+c_m_temp[1][1])
        roc_curve(tpr_batch_list,fpr_batch_list,c_m_temp,i)

    #performing online training
    for i in learning_rate:
        trained_weights=neuron_online_train(train_data, np.reshape(np.array([weights]),(3,1)), 100, target,i)
        print("The accuracy for onlie training for learning rate {} is: {}".format(i,prediction(test_data,trained_weights[0],target_test)[0]))
        print("The iterations are: {}".format(trained_weights[1]))
        print("The final weights are: {}".format(trained_weights[0]))
        c_m_temp = confusion_matrix(target_test,prediction(test_data,trained_weights[0],target_test)[1])
        tpr_batch_list[1] = c_m_temp[0][0]/(c_m_temp[0][0]+c_m_temp[1][0])
        fpr_batch_list[1] = c_m_temp[1][0]/(c_m_temp[1][0]+c_m_temp[1][1])
        roc_curve(tpr_batch_list,fpr_batch_list,c_m_temp,i)
