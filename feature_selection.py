import pandas as pd
from main import data_pre_processing, rename_data
from encoding import ordinal_encoding
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# read training dataset
trainingData = pd.read_csv("train.csv")
trainingDataRenamed = rename_data(trainingData)


# read prediction dataset
predictionData = pd.read_csv("test.csv")
predictionDataRenamed = rename_data(predictionData)


trainDataProcessed, predictionDataProcessed = data_pre_processing(trainingDataRenamed, predictionDataRenamed)
trainDataProcessed, predictionDataProcessed = ordinal_encoding(trainingDataRenamed, predictionDataRenamed)

X = trainDataProcessed.loc[:, trainDataProcessed.columns != 'total_income']
X = X.loc[:, X.columns != 'instance']
y = trainDataProcessed['total_income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(y_train.head())

clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)

feat_labels=['housing_situation', 'satisfaction', 'gender', 'hair_color', 'country', 'profession', 'degree', 'year_of_record', 'crime_level', 'work_experience', 'age', 'work_experience', 'city_size', 'height', 'additional_income']

for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)


sfm = SelectFromModel(clf, threshold=0.15)
sfm.fit(X_train, y_train)
for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])


