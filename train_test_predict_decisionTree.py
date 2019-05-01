import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz 

import pydot 

X = np.loadtxt('Heart_Disease_X.csv',skiprows=0, unpack=False, delimiter=',')
y = np.loadtxt('Heart_Disease_y.csv',skiprows=0, unpack=False, delimiter=',')

l = []

for i in range(len(X)):
    l.append(X[i])
    
X = np.array(l);

# split the data into a training sert and a testing set
X_train , X_test , y_train , y_test = train_test_split(X, y, random_state=42) 


print('Training...')
tree = DecisionTreeClassifier( random_state = 0 )
treePruned = DecisionTreeClassifier( max_depth = 8, random_state = 0 )
tree.fit (X_train, y_train ) 
treePruned.fit (X_train, y_train ) 

print('Testing...')
print("Accuracy on training set: {:.3f}".format(tree.score( X_train , y_train)))
print( "Accuracy on test set: {:.3f}".format(tree.score( X_test , y_test)))
print("Accuracy on training set with pruning: {:.3f}".format(treePruned.score( X_train , y_train)))
print( "Accuracy on test set with pruning: {:.3f}".format(treePruned.score( X_test , y_test)))

print( "Feature importances: \n {}".format (treePruned.feature_importances_ )) 

#Should predict healthy [0] and does
print('Predicting...')
X_new = np.array([ [22.0,1.0,4.0,130.0,110.0,0.0,0.0,180.0,0.0,0.3,2.0,0.0,3.0] ])
y_classification = treePruned.predict(X_new)
print('Prediction: {}->{}'.format(X_new[0], y_classification))

featureNames = ['age', 'sex','cp', 'thresBPS', 'chol', 'fbs', 'restECG', 'thalach', 'exang', 'oldPeak', 'slope', 'ca', 'thal']
# now create a bar chart and save it to a file
n_features = 13
plt.barh( range( n_features ), treePruned.feature_importances_ , align = 'center' ) 
plt.yticks( np.arange( n_features ), featureNames ) 
plt.xlabel( "Feature importance" ) 
plt.ylabel( "Feature" )
plt.ylim ( 1 , n_features ) 
plt.savefig(r"feature_importance.png",bbox_inches='tight')

export_graphviz(treePruned , out_file = "Heart_Disease_Tree.dot", class_names = [ "Healthy" , "Class I", "Class II", "Class III", "Class IV" ], feature_names = featureNames , impurity = False , filled = True ) 

(graph,) = pydot.graph_from_dot_file('./Heart_Disease_Tree.dot')
graph.write_png('./Heart_Disease_tree.png')

