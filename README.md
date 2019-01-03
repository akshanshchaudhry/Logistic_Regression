# Logistic_Regression
Implemented a perceptron for logistic regression in Python 3

• The data points for the 2 classes were generated randomly from multivariate normal distribution with u1=[1,0], u2=[0,1.5], sigma1=[[1 0.75][0.75 1]] and sigma2=[[1 0.75][0.75 1]] (where u1 and u2 are means and sigma1 and sigma2 are standard deviations).

• Used cross entropy as the objective funcntion ie. for the calculation of the error.

• Used gradient descent for weights updation

• Performed both Batch and Online training for the learning rates = {1,0.1,0.01}

• In the end, the ROC and area under the curve is generated (self written code).

• The accuracy of around 95% is achieved.


Technologies used:- Python, Pandas, Numpy, matplotlib
