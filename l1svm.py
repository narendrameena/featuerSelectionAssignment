# coding=utf-8

#author narendra
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.utils import check_random_state
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import svm
from pandas import DataFrame
from numpy.random import rand
from sklearn.utils import check_random_state
from sklearn import datasets


rnd = check_random_state(1)

data = load_svmlight_file("leu")

X_1 = data[0].todense().tolist()  # samples 72 features above 7192

y_1 = map(int,data[1])   # classes 2

#print(len(map(int,data[1])))
# set up dataset
n_samples = 72
n_features = 7000


#L1 SVM
l1svc = LinearSVC(penalty='l1', dual=False).fit(X_1, y_1)

model = SelectFromModel(l1svc, prefit=True)
X_2 = model.transform(X_1)
'''
# l2 data: non sparse, but less features
y_2 = np.sign(.5 - rnd.rand(n_samples))
X_2 = rnd.randn(n_samples, n_features / 5) + y_2[:, np.newaxis]
X_2 += 5 * rnd.randn(n_samples, n_features / 5)
'''
clf_sets = [(LinearSVC(penalty='l1', loss='squared_hinge', dual=False,
                       tol=1e-3),
             np.logspace(-2.3, -1.3, 10), X_1, y_1),
            (LinearSVC(penalty='l2', loss='squared_hinge', dual=True,
                       tol=1e-4),
             np.logspace(-4.5, -2, 10), X_2, y_1),
            (LinearSVC(penalty='l2', loss='squared_hinge', dual=True,
                       tol=1e-4),
             np.logspace(-4.5, -2, 10), X_1, y_1)]

colors = ['b', 'g', 'r', 'c']

for fignum, (clf, cs, X, y) in enumerate(clf_sets):
    # set up the plot for each regressor
    plt.figure(fignum, figsize=(9, 10))

    for k, train_size in enumerate(np.linspace(0.3, 0.7, 3)[::-1]):
        param_grid = dict(C=cs)
        # To get nice curve, we need a large number of iterations to
        # reduce the variance
        grid = GridSearchCV(clf, refit=False, param_grid=param_grid,
                            cv=ShuffleSplit(n=n_samples, train_size=train_size,
                                            n_iter=250, random_state=1))
        grid.fit(X, y)
        scores = [x[1] for x in grid.grid_scores_]

        scales = [(1, 'No scaling'),
                  ((n_samples * train_size), '1/n_samples'),
                  ]

        for subplotnum, (scaler, name) in enumerate(scales):
            plt.subplot(2, 1, subplotnum + 1)
            plt.xlabel('C')
            plt.ylabel('CV Score')
            grid_cs = cs * float(scaler)  # scale the C's
            plt.semilogx(grid_cs, scores, label="fraction %.2f" %
                         train_size)
            plt.title('scaling=%s, penalty=%s, loss=%s' %
                      (name, clf.penalty, clf.loss))

    plt.legend(loc="best")
plt.show()


#print(X_train.shape)
#L1 SVM
l1svc = LinearSVC(penalty='l1', dual=False).fit(X_1, y_1)
#l1svc
print(l1svc)




#L2 SVM trained on all the features
l2svc = LinearSVC(penalty='l2',dual=False).fit(X_1, y_1)
#print(l2svc)


#L2 SVM trained on the features selected by the L1 SVM
model = SelectFromModel(l1svc, prefit=True)
X_train_new = model.transform(X_train)
#Y_train_new = model.transform(y_train)
#print(X_train_new)
print(X_train_new.shape)
l2svc2 = LinearSVC(penalty='l2',dual=False).fit(X_train_new, y_train)
print(l2svc2)




#L2 SVM that uses RFE (with an L2-SVM) to select relevant features
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X_1, y_1)
ranking = rfe.ranking_.reshape(X_1[0].shape)


#L2 SVM that use the class RFECV which automatically selects the number of features

svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y_1, 2),
              scoring='accuracy')
rfecv.fit(X_1, y_1)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


#L1-SVM feature selection using subsamples

