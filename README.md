

# Model

使用 Scikit-learn 的 SVR 工具，一種基於 SVM 的 regression 模型。
由於 SVM 只能使用數字 feature，但 Sex 這項 feature 是 nominal 的資訊，
所以我將 Sex 轉換成 one-hot 的格式進行訓練。


# Evaluation

我使用三種指標來評斷 model 的效能，
分別是 Mean Absolute Error (MAE)，
Mean Square Error (MSE) 和 Root Mean Square Error (RMSE)，

我分別實驗了兩個版本，
原版(對照組)，去除 Sex 這項 feature 進行實驗，
one-hot 組，新增 Sex feature，並且將 nominal 轉為 one-hot 格式。

另外我還有比較 10-fold cross validation 和 單純切一份(10%)資料為 test dataset的差別。


# Experiment Result

請見 Experiment.md


# Conclusion

新增 Sex feature 後，整體 error 有下降，可見這項資訊對於分析 age 是個有用的資訊。
10-fold cross validation 可以得到更穩定的實驗結果，同時發現 10-fold 的版本 error 更低，
代表 data distribution 不是相當平均，單純取其中一部分進行訓練、測試，得到的結果會有偏差。

