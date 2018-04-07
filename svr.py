from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import pandas as pd
import math
import data_preprocessing


# original version
# data = pd.read_csv('data/data.txt')
# X = data[data.columns[1:8]]  # warning: missing 'Sex'
# y = data[data.columns[8:9]]

# one-hot version
data = data_preprocessing.convert_nominal('data/data.txt')  # convert nominal feature to one-hot feature
X = data[data.columns[:-1]]
y = data[data.columns[-1:]]

# 10-fold
kf = KFold(n_splits=10, shuffle=True)

mae_list, mse_list, rmse_list = [], [], []

for train_index, test_index in kf.split(X):
    print('{} {}'.format(len(train_index), len(test_index)))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # training
    clf = svm.SVR()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    mae_list.append(mean_absolute_error(y_test, y_pred))
    mse_list.append(mean_squared_error(y_test, y_pred))
    rmse_list.append(math.sqrt(mean_squared_error(y_test, y_pred)))

print(mae_list)
print(mse_list)
print(rmse_list)

print('Mean Absolute Error:    {:0.3f}'.format(sum(mae_list) / kf.get_n_splits()))
print('Mean Square Error:      {:0.3f}'.format(sum(mse_list) / kf.get_n_splits()))
print('Root Mean Square Error: {:0.3f}'.format(math.sqrt(sum(rmse_list) / kf.get_n_splits())))

