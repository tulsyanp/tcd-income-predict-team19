# Machine Learning Model to Predict Income

### The best algorithm using which the income is predicted is LIGHTGBM

### Required dependencies: 
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.
```
pandas, numpy, sklearn, lightgbm, categorical_encoders, matplotlib, seaborn
```

### Steps to Predict Output

1. In file ```predict.py```, uncomment the model that you want to use for prediction (By default, it runs random forest)
2. Run the command  ```python predict.py```
3. After a wait of few minutes the predicted result in generated in the file ```submission.csv```
4. To change model, uncomment following in ```predict.py``` and follow step 2.
```
model_linear_regression_predict(X, y, predictionDataFrame) # Linear Regression Model

model_random_forest_predict(X, y, predictionDataFrame) # Random Forest Model

model_xgboost_predict(X, y, predictionDataFrame) # XGBoost

model_lightbgm_predict(X, y, predictionDataFrame) # LightGBM
```


### Project Flow
1. Reads the dataset provided to train the model ```train.csv```
2. Removes the outliers from the training dataset
3. Reads the dataset on which income output is to be predicted ```test.csv```
4. Merge both the set set and pre-process data (remove outlier/target mean encoding)
5. Split the dataset into train and test dataset
6. Performs training of the model on the training dataset
7. Returns CSV for the prediction of income on the test dataset : ```submission.csv```

### Check Profile Score (Kaggle)
[click here!](https://www.kaggle.com/c/tcd-ml-comp-201920-income-pred-group/leaderboard)