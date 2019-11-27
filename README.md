# Machine Learning Model to Predict Income

### Rank #2 - The best algorithm using which the income is predicted is LIGHTGBM

## Things that worked
1. Creating a separate category or no category while using target encoding for categorical variables.
2. Creating a separating category for each special variables like: 0, '0', 'nA', #NUM!
3. Running 200000 iterations for lightGBM gave a score jump of around 200 - 300 points.
4. Target encoding for categorical variables
5. np.log target variable before target encoding and then encoding the categorical variables. Gave a jump of around 500 points.

### Required dependencies: 
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.
```
pandas, numpy, sklearn, lightgbm, categorical_encoders, matplotlib, seaborn
```

### Steps to Predict Output

1. In file ```predict.py```, uncomment the model that you want to use for prediction (By default, it runs LIGHTGBM)
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
2. Renames the column of training dataset
3. Reads the dataset on which income output is to be predicted ```test.csv```
4. Renames the column of training dataset
4. Pass both the dataset for preprocessing
5. After preprocessing, pass both the preprocessed dataset for target encoding
6. Performs training of the model on the training dataset using LIGHTGBM
7. Returns CSV for the prediction of income on the test dataset using ```sample-submission.csv``` : ```submission.csv```

### Check Profile Score (Kaggle)
[click here!](https://www.kaggle.com/c/tcd-ml-comp-201920-income-pred-group/leaderboard)