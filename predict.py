import pandas as pd
import numpy as np
from main import data_pre_processing, rename_data, remove_outliers, split_dataset,  remove_outlier_z_score, encoding
from model import model_linear_regression_predict, model_random_forest_predict, model_xgboost_predict, model_lightbgm_predict


# read training dataset
trainingData = pd.read_csv("train.csv")
trainingDataRenamed = rename_data(trainingData)
# trainDataProcessed = data_pre_processing(trainingDataRenamed)
# trainDataProcessed = remove_outliers(trainDataProcessed)

# trainDataFrame = remove_outliers(trainDataFrame)
# trainDataFrame = remove_outlier_z_score(trainDataFrame)
# print(trainDataFrame.dtypes)
# print(trainDataFrame.head())
# print(trainDataFrame.isnull().sum())
# print(trainDataFrame.shape)
# print(trainDataFrame['hair_color'].unique())


# read prediction dataset
predictionData = pd.read_csv("test.csv")
predictionDataRenamed = rename_data(predictionData)
# predictionDataProcessed = data_pre_processing(predictionDataRenamed)


trainDataProcessed, predictionDataProcessed = data_pre_processing(trainingDataRenamed, predictionDataRenamed)

# trainDataProcessed = remove_outliers(trainDataProcessed)

print("1")

# encoding data
trainDataFrame, predictionDataFrame = encoding(trainDataProcessed, predictionDataProcessed)


# # data pre processing
# preProcessedDataFrame = data_pre_processing(trainingDataRenamed, predictionDataRenamed)
# # split dataset
# trainDataFrame, predictionDataFrame = split_dataset(preProcessedDataFrame)
# print("1")


# model training by dividing target and attributes
X = trainDataFrame.loc[:, trainDataFrame.columns != 'total_income']
y = trainDataFrame['total_income']

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
