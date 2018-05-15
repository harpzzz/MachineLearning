
import  numpy as np
from sklearn.datasets import load_iris
from sklearn import tree


dict = {0:"setosa",1:"versicolor",2:"virginica"}

iris = load_iris()


# name of the  features
print(iris.feature_names)


# name of the label
print(iris.target_names)

#TestingIndex
testIndex = [0,125,75,90, 24]



#train data
train_label = np.delete(iris.target,testIndex)
train_features = np.delete(iris.data,testIndex,axis=0)


#testData
test_label = iris.target[testIndex]
test_features = iris.data[testIndex]


#Classifier
clf = tree.DecisionTreeClassifier()


#train the classifier
clf.fit(train_features,train_label)





#test data
print(test_label)

# prediction result
print(clf.predict(test_features))

