from sklearn import tree

# supervisor learning

# Classifier
feature = [[140,0],[130,0], [150,1],[170,1]]
labels = ["apple","apple","orange","orange"]


# We use Decicion tree
clf = tree.DecisionTreeClassifier()

#train the classifier
clf = clf.fit(feature,labels)


# predict
print(clf.predict([[160,1]]))

