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
        cols=['housing_situation', 'satisfaction', 'gender', 'hair_color', 'country', 'profession', 'degree']
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


def target_mean_encoding(trainData, predictionData, target):

    oEncoder = ce.TargetEncoder(
        cols=['housing_situation', 'satisfaction', 'gender', 'hair_color', 'country', 'profession', 'degree']
    )

    oEncoder.fit(trainData, target)
    trainDataFrame = oEncoder.transform(trainData)
    predictionDataFrame = oEncoder.transform(predictionData)

    return trainDataFrame, predictionDataFrame


def leave_one_out_encoding(trainData, predictionData):

    oEncoder = ce.LeaveOneOutEncoder(
        cols=['country', 'profession', 'degree']
    )
    target = trainData['total_income']

    oEncoder.fit(trainData, target)
    trainDataFrame = oEncoder.transform(trainData)
    predictionDataFrame = oEncoder.transform(predictionData)

    return trainDataFrame, predictionDataFrame
