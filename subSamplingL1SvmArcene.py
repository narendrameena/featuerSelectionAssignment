import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
import random
from scipy.sparse import csr_matrix
import csv


#data
#data = load_svmlight_file("leu")

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



l =len(data[1])
start = int(round(l*0.70,0))

#N = random.sample(range(start,l), 1)
N = int(round(l*0.80,0))
print("Number of sub Samples %d" %N)
i = np.random.choice(np.arange(data[0].shape[0]), N, replace=False)


sub_data = data[0][i.tolist()]
sub_sample = data[1][i.tolist()] # check for this step


X_1 = sub_data.todense().tolist()  # samples 72 features above 7129
y_1 = map(int,sub_sample)   # classes 2


#L1 SVM
l1svc = LinearSVC(penalty='l1', dual=False).fit(X_1, y_1)

#print(len(l1svc.coef_[0]))





coef = l1svc.coef_.tolist()[0]
#print(coef[0])
#print(l1svc.coef_.tolist()[0])
#print[i for i, j in enumerate(coef) if j > 0]

#print(len(l1svc.coef_.tolist()[0]))

print("Number of features have non-zero weight vector coefficients %d " %sum(1 for i in coef if i  != 0))

#For each feature compute a score that is the number of sub-samples for which that feature yielded a non-zero weight vector coefficient
'''
sampleListCoef = []
print(len(l1svc.coef_[0].tolist()))
for k in range(0,len(l1svc.coef_[0].tolist())):
    for j in range(start,l):
        i = np.random.choice(np.arange(data[0].shape[0]), j, replace=False)
        sub_data = data[0][i.tolist()]
        sub_sample = data[1][i.tolist()] # check for this step
        X_1 = sub_data.todense().tolist()  # samples 72 features above 7129
        y_1 = map(int,sub_sample)   # classes 2
        #L1 SVM
        l1svc = LinearSVC(penalty='l1', dual=False).fit(X_1, y_1)
        coef = map(int,np.asarray(l1svc.coef_[0]))
        if(coef[k] > 0):
            sampleListCoef.append[j]
        else:
            sampleListCoef + [0]


print("number of sub-samples for which that feature yielded a non-zero weight vector coefficient :")
print(sampleListCoef)
'''