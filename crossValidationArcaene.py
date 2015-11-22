import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import csv

#data
#data = load_svmlight_file("leu")
#X_1 = data[0].todense().tolist()  # samples 72 features above 7192
#y_1 = map(int,data[1])   # classes 2

f = open("arcene_train.data") # read the data file
f1 = open("arcene_train.labels") # read the lable file
try:
    a = []
    for row in csv.reader(f):
        row = [int(i) for i in row[0].split()]
        a.append(row)   # data matrix
    b = []
    for row in csv.reader(f1):
        row = map(int,row)
        b.append(row[0])   # labels matrix
        #print(golub(a,b))
finally:
    f.close   # close the files
    f1.close


a =csr_matrix(a) # convert into sparse matrix
b = np.asarray(b)


data =  (a,b)
#print(a.shape)
#print(b.shape)
#print(data[0].shape)
#print(data[1].shape)

c = 1.9 #SVM soft-margin constant

print("SVM soft-margin constant %d", c)

#test_size= 0.4 # selecting number of samples

X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[0],data[1], test_size=0.4, random_state=1)

#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)

#L1 SVM
clf = LinearSVC(penalty='l1', dual=False,C =c)
scores = cross_validation.cross_val_score(clf, data[0], data[1], cv=10)

print(scores)
print("L1 SVM \n  Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#L2 SVM trained on all the features
clf = LinearSVC(penalty='l2',dual=False,C= c)
scores = cross_validation.cross_val_score(clf, data[0], data[1], cv=10)

print(scores)
print("L2 SVM trained on all the features \n Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#L2 SVM trained on the features selected by the L1 SVM
clf = LinearSVC(penalty='l1', dual=False, C=c).fit(data[0], data[1])
model = SelectFromModel(clf, prefit=True)
X = model.transform(data[0])

#print(X.shape)
clf = LinearSVC(penalty='l2',dual=False, C=c)
scores = cross_validation.cross_val_score(clf, X, data[1], cv=10)

print(scores)
print("L2 SVM trained on the features selected by the L1 SVM. \n  Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




#L2 SVM that use the class RFECV which automatically selects the number of features

clf = LinearSVC(penalty='l2',dual=False,C=c)
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(data[1], 2),scoring='accuracy')
rfecv.fit(data[0], data[1])
#scores = cross_validation.cross_val_score(rfecv, data[0], data[1], cv=10)
print("Optimal number of features : %d" % rfecv.n_features_)
scores = rfecv.grid_scores_
print(scores)
print("L2 SVM that use the class RFECV which automatically selects the number of features. \n Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
'''

#L2 SVM that uses RFE (with an L2-SVM) to select relevant features
clf = LinearSVC(penalty='l2',dual=False,C =c)
rfe = RFE(estimator=clf, n_features_to_select=10, step=1)
rfe.fit(data[0], data[1])
scores = cross_validation.cross_val_score(rfe.estimator_ , data[0], data[1], cv=10)
print(scores)
print("L2 SVM that uses RFE (with an L2-SVM) to select relevant features. \n Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

