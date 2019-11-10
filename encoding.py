import category_encoders as ce
from feature_engine.categorical_encoders import OneHotCategoricalEncoder


def one_hot_encoder(trainData, predictionData):
    encoder = OneHotCategoricalEncoder(
        top_categories=None,
        variables=['housing_situation', 'satisfaction', 'gender', 'hair_color'],
        drop_last=True)

    encoder.fit(trainData)
    trainDataFrame = encoder.transform(trainData)
    predictionDataFrame = encoder.transform(predictionData)

    return trainDataFrame, predictionDataFrame


def ordinal_encoding(trainData, predictionData):

    oEncoder = ce.OrdinalEncoder(
        cols=['housing_situation', 'satisfaction', 'gender', 'hair_color']
    )

    oEncoder.fit(trainData)
    trainDataFrame = oEncoder.transform(trainData)
    predictionDataFrame = oEncoder.transform(predictionData)

    return trainDataFrame, predictionDataFrame


def hash_encoding(trainData, predictionData):

    oEncoder = ce.HashingEncoder(
        cols=['housing_situation', 'satisfaction', 'gender', 'hair_color', 'country', 'profession', 'degree']
    )

    oEncoder.fit(trainData)
    trainDataFrame = oEncoder.transform(trainData)
    predictionDataFrame = oEncoder.transform(predictionData)

    return trainDataFrame, predictionDataFrame


def target_mean_encoding(trainData, predictionData):

    # cat_vars = ['country', 'profession', 'degree']
    #
    # code_map = dict()
    # default_map = dict()
    #
    # for v in cat_vars:
    #     prior = trainData['total_income'].mean()
    #     n = trainData.groupby(v).size()
    #     mu = trainData.groupby(v)['total_income'].mean()
    #     # mu_smoothed = (n * mu + 10 * prior) / (n + 10)
    #
    #     trainData.loc[:, v] = trainData[v].map(mu)
    #     code_map[v] = mu
    #     default_map[v] = prior
    #
    # for v in cat_vars:
    #     predictionData.loc[:, v] = predictionData[v].map(code_map[v])
    #
    # trainDataFrame = trainData
    # predictionDataFrame = predictionData

    oEncoder = ce.TargetEncoder(
        cols=['country', 'profession', 'degree'],
        smoothing=10
    )
    target = trainData['total_income']

    oEncoder.fit(trainData, target)
    trainDataFrame = oEncoder.transform(trainData)
    predictionDataFrame = oEncoder.transform(predictionData)

    return trainDataFrame, predictionDataFrame
