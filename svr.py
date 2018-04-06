from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd


data = pd.read_csv('data/data.txt')

X = data[data.columns[1:8]]  # warning: missing 'Sex'
y = data[data.columns[8:9]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=15)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

clf = svm.SVR()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(mse)

