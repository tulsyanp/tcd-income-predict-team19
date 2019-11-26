import pandas as pd
import numpy as np
from main import data_pre_processing, rename_data, remove_outliers, split_dataset,  remove_outlier_z_score, encoding
from model import model_linear_regression_predict, model_random_forest_predict, model_xgboost_predict, model_lightbgm_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# read training dataset
trainingData = pd.read_csv("train.csv")
trainingDataRenamed = rename_data(trainingData)
target = np.log(trainingDataRenamed['total_income'])
trainingDataRenamed = trainingDataRenamed.loc[:, trainingDataRenamed.columns != 'total_income']

# print(trainDataFrame.dtypes)
# print(trainDataFrame.head())
# print(trainDataFrame.isnull().sum())
# print(trainDataFrame.shape)
# print(trainDataFrame['hair_color'].unique())
# print(trainData['hair_color'].value_counts())


# read prediction dataset
predictionData = pd.read_csv("test.csv")
predictionDataRenamed = rename_data(predictionData)
predictionDataRenamed = predictionDataRenamed.loc[:, predictionDataRenamed.columns != 'total_income']


trainDataProcessed, predictionDataProcessed = data_pre_processing(trainingDataRenamed, predictionDataRenamed)

print("1")

# encoding data
trainDataFrame, predictionDataFrame = encoding(trainDataProcessed, predictionDataProcessed, target)


# model training by dividing target and attributes
X = trainDataFrame
y = target

print("2")
# Uncomment to predict using Linear Regression Model
# model_linear_regression_predict(X, y, predictionDataFrame)
# print("3")

# Uncomment to predict using Random Forest Model
# model_random_forest_predict(X, y, predictionDataFrame)

# Uncomment to predict using XGBoost
# model_xgboost_predict(X, y, predictionDataFrame)

# Uncomment to predict using XGBoost
model_lightbgm_predict(X, y, predictionDataFrame)
