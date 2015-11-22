# coding=utf-8


from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file

#data
data = load_svmlight_file("leu")

X_1 = data[0].todense().tolist()  # samples 72 features above 7192

y_1 = map(int,data[1])   # classes 2

lr = LinearSVC(penalty='l1', dual=False).fit(X_1, y_1)

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validated:
predicted = cross_val_predict(lr, X_1, y_1, cv=10)

fig, ax = plt.subplots()
ax.scatter(y_1, predicted)
ax.plot([min(y_1), max(y_1)], [min(y_1), max(y_1)], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
