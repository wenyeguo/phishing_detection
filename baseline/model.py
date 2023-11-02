import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from feature_extraction import read_data, read_features

URL, Y = read_data()
# print(Y.shape)
X = read_features("features/features.csv")
# print(X.shape)

# clf = GaussianNB()
clf = LogisticRegression(solver='lbfgs', max_iter=1000)
# clf = RandomForestClassifier()

scores = cross_val_score(clf, X, Y,cv=5)
# print("GaussianNB clf, 5-fold cross validation")
print("LogisticRegression clf, 5-fold cross validation")
# print("RandomForest clf, 5-fold cross validation")

print("Accuracy", scores)
print(f"Average accuracy = {np.mean(scores):.3f}")
print(f"Average Precision = {np.mean(cross_val_score(clf,X,Y,cv=5,scoring='precision')):.3f}")
print(f"Average Recall = {np.mean(cross_val_score(clf,X,Y,cv=5,scoring='recall')):.3f}")
print(f"Average F1 = {np.mean(cross_val_score(clf,X,Y,cv=5,scoring='f1')):.3f}")

