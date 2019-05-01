import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
 

X = np.loadtxt('Heart_Disease_X.csv',skiprows=0, unpack=False, delimiter=',')
y = np.loadtxt('Heart_Disease_y.csv',skiprows=0, unpack=False, delimiter=',')

X_train , X_test , y_train , y_test = train_test_split(X, y, random_state=42) 

# compute the minimum value per feature on the training set 
min_on_training=X_train.min(axis=0) 
range_on_training = (X_train - min_on_training).max(axis = 0) 


X_train_scaled = ( X_train - min_on_training ) / range_on_training 
X_test_scaled = ( X_test - min_on_training ) / range_on_training 

print('Training...')
svc = SVC(kernel='rbf', C=725, gamma='auto') # change c to 10 for better fit
svc.fit(X_train_scaled, y_train) 

print('Testing...')
print( "Accuracy on training set: {:.2f}".format(svc.score(X_train_scaled, y_train))) 
print( "Accuracy on test set: {:.2f}".format(svc.score(X_test_scaled, y_test ))) 

#Should predict healthy [0] and does
print('Predicting...')
X_new = np.array([ [22.0,1.0,4.0,130.0,110.0,0.0,0.0,180.0,0.0,0.3,2.0,0.0,3.0] ])
y_classification = svc.predict(X_new)
print('Prediction: {}->{}'.format(X_new[0], y_classification))

featureNames = ['age', 'sex','cp', 'thresBPS', 'chol', 'fbs', 'restECG', 'thalach', 'exang', 'oldPeak', 'slope', 'ca', 'thal']
n_features = 13
plt.cla()
plt.clf()
plt.boxplot( X_train_scaled , manage_xticks = True ) 
plt.yscale( "symlog" ) 
plt.xlabel( "Feature index" ) 
plt.xlim(1, n_features)
plt.ylabel ( "Feature magnitude" ) 
plt.savefig(r"feature_magnitude.png",bbox_inches='tight')
