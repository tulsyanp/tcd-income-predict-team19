import numpy as np
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

from encoding import one_hot_encoder, target_mean_encoding, ordinal_encoding, leave_one_out_encoding
from preprocess import additional_income, housing_situation, work_experience, year_of_record, satisfaction, gender, \
    age, country, profession, degree, hair_color


def rename_data(data):
    data.rename(columns={'Instance': 'instance'}, inplace=True)
    data.rename(columns={'Year of Record': 'year_of_record'}, inplace=True)
    data.rename(columns={'Housing Situation': 'housing_situation'}, inplace=True)
    data.rename(columns={'Crime Level in the City of Employement': 'crime_level'}, inplace=True)
    data.rename(columns={'Work Experience in Current Job [years]': 'work_experience'}, inplace=True)
    data.rename(columns={'Satisfation with employer': 'satisfaction'}, inplace=True)
    data.rename(columns={'Gender': 'gender'}, inplace=True)
    data.rename(columns={'Age': 'age'}, inplace=True)
    data.rename(columns={'Country': 'country'}, inplace=True)
    data.rename(columns={'Size of City': 'city_size'}, inplace=True)
    data.rename(columns={'Profession': 'profession'}, inplace=True)
    data.rename(columns={'University Degree': 'degree'}, inplace=True)
    data.rename(columns={'Wears Glasses': 'glasses'}, inplace=True)
    data.rename(columns={'Hair Color': 'hair_color'}, inplace=True)
    data.rename(columns={'Body Height [cm]': 'height'}, inplace=True)
    data.rename(columns={'Yearly Income in addition to Salary (e.g. Rental Income)': 'additional_income'}, inplace=True)
    data.rename(columns={'Total Yearly Income [EUR]': 'total_income'}, inplace=True)

    return data


def data_pre_processing(trainData, predictionData):

    # dataFrame = dataFrame.drop(columns=['crime_level', 'city_size', 'glasses', 'height'])

    trainData = trainData.drop(columns=['instance'])
    predictionData = predictionData.drop(columns=['instance'])

    trainData, predictionData = additional_income(trainData, predictionData)

    trainData, predictionData = housing_situation(trainData, predictionData)

    trainData, predictionData = work_experience(trainData, predictionData)

    trainData, predictionData = year_of_record(trainData, predictionData)

    trainData, predictionData = satisfaction(trainData, predictionData)

    trainData, predictionData = gender(trainData, predictionData)

    trainData, predictionData = age(trainData, predictionData)

    trainData, predictionData = country(trainData, predictionData)

    trainData, predictionData = profession(trainData, predictionData)

    trainData, predictionData = degree(trainData, predictionData)

    trainData, predictionData = hair_color(trainData, predictionData)

    return trainData, predictionData


def encoding(trainData, predictionData, target):

    # print(predictionData.shape)
    #
    # trainDataFrame, predictionDataFrame = one_hot_encoder(trainData, predictionData)
    #
    # print(predictionDataFrame.shape)

    trainDataFrame, predictionData = target_mean_encoding(trainData, predictionData, target)

    print(trainDataFrame.shape)

    return trainDataFrame, predictionData


def remove_outliers(data):
    data.drop(data[data['total_income'] < 0].index, inplace=True)
    # data['total_income'].abs()
    # data.drop(data['Income'].idxmax(), inplace=True)
    # data.drop_duplicates(inplace=True, keep=False)

    return data


def remove_outlier_z_score(data):
    # outliers = []
    #
    # threshold = 3
    # mean_1 = np.mean(data['total_income'])
    # std_1 = np.std(data['total_income'])
    #
    # for y in data['total_income']:
    #     z_score = (y - mean_1) / std_1
    #     if np.abs(z_score) > threshold:
    #         # outliers.append(y)
    #         data.drop(data[data['total_income'] == y].index, inplace=True)

    z = np.abs(stats.zscore(data['total_income']))
    data['z_score'] = z
    data.drop(data[data['z_score'] < 3].index, inplace=True)
    data = data.drop(columns=['z_score'])

    return data


def split_dataset(data):
    mask = data['total_income'] > 0
    trainDataFrame = data[mask]
    predictionDataFrame = data[~mask]

    return trainDataFrame, predictionDataFrame


def heatmap(data):
    corr = data.corr()
    ax = sns.heatmap(
        corr,
        vmin=0, vmax=1, center=0.5,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        linewidths=.5
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.show(ax)
