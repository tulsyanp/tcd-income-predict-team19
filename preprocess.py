import pandas as pd
import numpy as np
import math


def additional_income(trainData, predictionData):
    trainData['additional_income'] = trainData['additional_income'].replace({'EUR': ''}, regex=True).astype(float)
    predictionData['additional_income'] = predictionData['additional_income'].replace({'EUR': ''}, regex=True).astype(float)

    return trainData, predictionData


def housing_situation(trainData, predictionData):
    trainData['housing_situation'] = trainData['housing_situation'].str.lower()
    predictionData['housing_situation'] = predictionData['housing_situation'].str.lower()

    maximum_occ = trainData['housing_situation'].mode().iloc[0]

    trainData['housing_situation'].replace(['nA', 0, '0'], [maximum_occ, 'missing', 'missing'], inplace=True)
    trainData['housing_situation'].fillna(maximum_occ, inplace=True)
    trainData['housing_situation'] = trainData['housing_situation'].str.strip()

    predictionData['housing_situation'].replace(['nA', 0, '0'], [maximum_occ, 'missing', 'missing'], inplace=True)
    predictionData['housing_situation'].fillna(maximum_occ, inplace=True)
    predictionData['housing_situation'] = predictionData['housing_situation'].str.strip()

    return trainData, predictionData


def work_experience(trainData, predictionData):
    trainData['work_experience'].replace(['#NUM!'], [np.nan], inplace=True)
    trainData['work_experience'] = pd.to_numeric(trainData['work_experience'])

    predictionData['work_experience'].replace(['#NUM!'], [np.nan], inplace=True)
    predictionData['work_experience'] = pd.to_numeric(predictionData['work_experience'])

    mean_value = round(trainData['work_experience'].mean())

    trainData['work_experience'].fillna(mean_value, inplace=True)
    predictionData['work_experience'].fillna(mean_value, inplace=True)

    return trainData, predictionData


def year_of_record(trainData, predictionData):
    median_value = round(trainData['year_of_record'].median())

    trainData['year_of_record'].fillna(median_value, inplace=True)
    predictionData['year_of_record'].fillna(median_value, inplace=True)

    trainData['year_of_record'] = trainData['year_of_record'] ** (1 / 2)
    predictionData['year_of_record'] = predictionData['year_of_record'] ** (1 / 2)

    return trainData, predictionData


def satisfaction(trainData, predictionData):
    trainData['satisfaction'] = trainData['satisfaction'].str.lower()
    predictionData['satisfaction'] = predictionData['satisfaction'].str.lower()

    maximum_occ = trainData['satisfaction'].mode().iloc[0]

    trainData['satisfaction'].fillna(maximum_occ, inplace=True)
    trainData['satisfaction'] = trainData['satisfaction'].str.strip()

    predictionData['satisfaction'].fillna(maximum_occ, inplace=True)
    predictionData['satisfaction'] = predictionData['satisfaction'].str.strip()

    return trainData, predictionData


def gender(trainData, predictionData):
    trainData['gender'] = trainData['gender'].str.lower()
    predictionData['gender'] = predictionData['gender'].str.lower()

    maximum_occ = trainData['gender'].mode().iloc[0]

    trainData['gender'].fillna(maximum_occ, inplace=True)
    trainData['gender'].replace(['0', 'f'], ['missing', 'female'], inplace=True)
    trainData['gender'] = trainData['gender'].str.strip()

    predictionData['gender'].fillna(maximum_occ, inplace=True)
    predictionData['gender'].replace(['0', 'f'], ['missing', 'female'], inplace=True)
    predictionData['gender'] = predictionData['gender'].str.strip()

    return trainData, predictionData


def age(trainData, predictionData):
    mean_value = round(trainData['age'].mean())

    trainData['age'].fillna(mean_value, inplace=True)
    trainData['age'] = trainData['age'] ** (1 / 2)

    predictionData['age'].fillna(mean_value, inplace=True)
    predictionData['age'] = predictionData['age'] ** (1 / 2)

    return trainData, predictionData


def country(trainData, predictionData):
    trainData['country'] = trainData['country'].str.lower()
    predictionData['country'] = predictionData['country'].str.lower()

    maximum_occ = trainData['country'].mode().iloc[0]

    trainData['country'].fillna(maximum_occ, inplace=True)
    trainData['country'] = trainData['country'].str.strip()

    predictionData['country'].fillna(maximum_occ, inplace=True)
    predictionData['country'] = predictionData['country'].str.strip()

    return trainData, predictionData


def profession(trainData, predictionData):
    trainData['profession'] = trainData['profession'].str.lower()
    predictionData['profession'] = predictionData['profession'].str.lower()

    maximum_occ = trainData['profession'].mode().iloc[0]

    trainData['profession'].fillna(maximum_occ, inplace=True)
    trainData['profession'] = trainData['profession'].str.strip()

    predictionData['profession'].fillna(maximum_occ, inplace=True)
    predictionData['profession'] = predictionData['profession'].str.strip()

    # dataFrame['profession'] = dataFrame['profession'].str.slice(start=0, stop=3)

    return trainData, predictionData


def degree(trainData, predictionData):
    trainData['degree'] = trainData['degree'].str.lower()
    predictionData['degree'] = predictionData['degree'].str.lower()

    maximum_occ = trainData['degree'].mode().iloc[0]

    trainData['degree'].fillna(maximum_occ, inplace=True)
    trainData['degree'].replace(['0'], ['missing'], inplace=True)
    trainData['degree'] = trainData['degree'].str.strip()

    predictionData['degree'].fillna(maximum_occ, inplace=True)
    predictionData['degree'].replace(['0'], ['missing'], inplace=True)
    predictionData['degree'] = predictionData['degree'].str.strip()

    return trainData, predictionData


def hair_color(trainData, predictionData):
    trainData['hair_color'] = trainData['hair_color'].str.lower()
    predictionData['hair_color'] = predictionData['hair_color'].str.lower()

    maximum_occ = trainData['hair_color'].mode().iloc[0]

    trainData['hair_color'].fillna(maximum_occ, inplace=True)
    trainData['hair_color'].replace(['0', 'unknown'], ['missing', 'missing'], inplace=True)
    trainData['hair_color'] = trainData['hair_color'].str.strip()

    predictionData['hair_color'].fillna(maximum_occ, inplace=True)
    predictionData['hair_color'].replace(['0', 'unknown'], ['missing', 'missing'], inplace=True)
    predictionData['hair_color'] = predictionData['hair_color'].str.strip()

    return trainData, predictionData