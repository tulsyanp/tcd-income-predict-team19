import pandas as pd
import numpy as np
import math


def additional_income(trainData, predictionData):
    trainData['additional_income'] = trainData['additional_income'].replace(regex=['EUR$'], value='')
    trainData['additional_income'] = pd.to_numeric(trainData['additional_income'])

    predictionData['additional_income'] = predictionData['additional_income'].replace(regex=['EUR$'], value='')
    predictionData['additional_income'] = pd.to_numeric(predictionData['additional_income'])

    return trainData, predictionData


def housing_situation(trainData, predictionData):
    trainData['housing_situation'] = trainData['housing_situation'].str.lower()
    predictionData['housing_situation'] = predictionData['housing_situation'].str.lower()

    trainData['housing_situation'].replace(['na', 0, '0'], ['missing1', 'missing2', 'missing3'], inplace=True)
    # trainData['housing_situation'].fillna(method='bfill', inplace=True)
    # trainData['housing_situation'] = trainData['housing_situation'].str.partition(' ')[0]  # try
    trainData['housing_situation'] = trainData['housing_situation'].str.strip()

    predictionData['housing_situation'].replace(['na', 0, '0'], ['missing1', 'missing2', 'missing3'], inplace=True)
    # predictionData['housing_situation'].fillna(method='bfill', inplace=True)
    # predictionData['housing_situation'] = predictionData['housing_situation'].str.partition(' ')[0]  # try
    predictionData['housing_situation'] = predictionData['housing_situation'].str.strip()

    return trainData, predictionData


def work_experience(trainData, predictionData):

    trainData['work_experience'].replace(['#NUM!'], [np.nan], inplace=True)
    trainData['work_experience'] = pd.to_numeric(trainData['work_experience'])

    predictionData['work_experience'].replace(['#NUM!'], [np.nan], inplace=True)
    predictionData['work_experience'] = pd.to_numeric(predictionData['work_experience'])

    mean_value = round(trainData['work_experience'].mean())

    trainData['work_experience'].fillna(method='ffill', inplace=True)
    predictionData['work_experience'].fillna(method='ffill', inplace=True)

    return trainData, predictionData


def year_of_record(trainData, predictionData):
    median_value = round(trainData['year_of_record'].median())

    trainData['year_of_record'].fillna(1942.0, inplace=True)  # to try something else
    predictionData['year_of_record'].fillna(1942.0, inplace=True)  # to ask group member

    return trainData, predictionData


def satisfaction(trainData, predictionData):
    trainData['satisfaction'] = trainData['satisfaction'].str.lower()
    predictionData['satisfaction'] = predictionData['satisfaction'].str.lower()

    maximum_occ = trainData['satisfaction'].mode().iloc[0]

    # trainData['satisfaction'].fillna(method='ffill', inplace=True) # to try maximum_occ
    # trainData['satisfaction'] = trainData['satisfaction'].str.partition(' ')[0]  # try
    trainData['satisfaction'] = trainData['satisfaction'].str.strip()

    # predictionData['satisfaction'].fillna(method='ffill', inplace=True)
    # predictionData['satisfaction'] = predictionData['satisfaction'].str.partition(' ')[0]  # try
    predictionData['satisfaction'] = predictionData['satisfaction'].str.strip()

    return trainData, predictionData


def gender(trainData, predictionData):
    trainData['gender'] = trainData['gender'].str.lower()
    predictionData['gender'] = predictionData['gender'].str.lower()

    maximum_occ = trainData['gender'].mode().iloc[0] # 0 = male, f = female, nan = mode or ffill

    # trainData['gender'].fillna(method='ffill', inplace=True) # try with maximum_occ
    trainData['gender'].replace(['0', 'f'], ['missing', 'female'], inplace=True)  # to try
    trainData['gender'] = trainData['gender'].str.strip()

    # predictionData['gender'].fillna(method='ffill', inplace=True)
    predictionData['gender'].replace(['0', 'f'], ['missing', 'female'], inplace=True)
    predictionData['gender'] = predictionData['gender'].str.strip()

    return trainData, predictionData


def age(trainData, predictionData):

    # all correct value

    return trainData, predictionData


def country(trainData, predictionData):
    trainData['country'] = trainData['country'].str.lower()
    predictionData['country'] = predictionData['country'].str.lower()

    # trainData['country'].replace(['0'], ['honduras'], inplace=True)  # to try with unknown
    # trainData['country'] = trainData['country'].str.strip()
    #
    # predictionData['country'].fillna('honduras', inplace=True)  # to try with unknown
    # predictionData['country'] = predictionData['country'].str.strip()  # only one test, 0 & 0.0 = mode or unknown

    return trainData, predictionData


def profession(trainData, predictionData):
    trainData['profession'] = trainData['profession'].str.lower()
    predictionData['profession'] = predictionData['profession'].str.lower()

    maximum_occ = trainData['profession'].mode().iloc[0] # to try with virtual systems engineer or bfill and ffill

    # trainData['profession'].fillna(maximum_occ, inplace=True)
    trainData['profession'] = trainData['profession'].str.slice(start=0, stop=3)
    trainData['profession'] = trainData['profession'].str.strip()

    # predictionData['profession'].fillna(maximum_occ, inplace=True)
    predictionData['profession'] = predictionData['profession'].str.slice(start=0, stop=3)
    predictionData['profession'] = predictionData['profession'].str.strip()

    return trainData, predictionData


def degree(trainData, predictionData):
    trainData['degree'] = trainData['degree'].str.lower()
    predictionData['degree'] = predictionData['degree'].str.lower()

    maximum_occ = trainData['degree'].mode().iloc[0]

    # trainData['degree'].fillna(method='ffill', inplace=True) # try with maximum_occ
    trainData['degree'].replace(['0'], ['missing'], inplace=True) # missing
    trainData['degree'] = trainData['degree'].str.strip()

    # predictionData['degree'].fillna(method='ffill', inplace=True)
    predictionData['degree'].replace(['0'], ['missing'], inplace=True)
    predictionData['degree'] = predictionData['degree'].str.strip()

    return trainData, predictionData


def hair_color(trainData, predictionData):
    trainData['hair_color'] = trainData['hair_color'].str.lower()
    predictionData['hair_color'] = predictionData['hair_color'].str.lower()

    maximum_occ = trainData['hair_color'].mode().iloc[0] # ffill or bfill

    # trainData['hair_color'].fillna(method='ffill', inplace=True) # try with maximum_occ
    trainData['hair_color'].replace(['0'], ['missing'], inplace=True)
    trainData['hair_color'] = trainData['hair_color'].str.strip()

    # predictionData['hair_color'].fillna(method='ffill', inplace=True)
    predictionData['hair_color'].replace(['0'], ['missing'], inplace=True)
    predictionData['hair_color'] = predictionData['hair_color'].str.strip()

    return trainData, predictionData