import pandas as pd
import numpy as np
from main import data_pre_processing, rename_data, remove_outliers
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# read training dataset
trainingData = pd.read_csv("train.csv")
trainingDataRenamed = rename_data(trainingData)


# read prediction dataset
predictionData = pd.read_csv("test.csv")
predictionDataRenamed = rename_data(predictionData)


trainDataProcessed, predictionDataProcessed = data_pre_processing(trainingDataRenamed, predictionDataRenamed)

trainDataProcessed = remove_outliers(trainDataProcessed)


