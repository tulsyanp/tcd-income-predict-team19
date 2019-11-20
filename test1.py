import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import ensemble
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import lightgbm as lgb



#Reading training and testing data
dataset=pd.read_csv('train.csv')  #training data
dataset1=pd.read_csv('test.csv')  #testing data
dataset2=pd.read_csv('submission-sample.csv') #submission file

#for col in dataset.columns:
#    print(col)

dataset['Gender'] = dataset['Gender'].replace('f','female')
dataset['Gender'] = dataset['Gender'].fillna('male')
dataset['Gender'].replace(['0'], ['missing'], inplace=True)

dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].replace(regex=['EUR$'], value='')
dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = pd.to_numeric(dataset['Yearly Income in addition to Salary (e.g. Rental Income)'])

dataset['Work Experience in Current Job [years]'].replace(['#NUM!'], [np.nan], inplace=True)
dataset['Work Experience in Current Job [years]'] = pd.to_numeric(dataset['Work Experience in Current Job [years]'])
dataset['Work Experience in Current Job [years]'].fillna(method="ffill", inplace=True)

dataset['Satisfation with employer'] = dataset['Satisfation with employer'].fillna('Average')

dataset['Profession'] = dataset['Profession'].fillna('virtual systems engineer')
dataset['Profession'] = dataset['Profession'].str.slice(start=0, stop=3)
dataset['Profession'] = dataset['Profession'].str.strip()

dataset['University Degree'] = dataset['University Degree'].fillna('Bachelor')
dataset['University Degree'].replace(['0'], ['missing'], inplace=True)

dataset['Hair Color'] = dataset['Hair Color'].fillna('Black')
dataset['Hair Color'].replace(['0'], ['missing'], inplace=True)

dataset['Year of Record'] = dataset['Year of Record'].fillna(1942.0)

dataset['Housing Situation'].replace(['nA', 0, '0'], ['missing', 'missing1', 'missing2'], inplace=True)
dataset['Housing Situation'].fillna('missing3', inplace=True)

X = dataset[['Gender', 'University Degree','Hair Color','Profession','Country','Satisfation with employer','Wears Glasses','Housing Situation','Year of Record','Crime Level in the City of Employement','Work Experience in Current Job [years]','Age','Size of City','Body Height [cm]','Yearly Income in addition to Salary (e.g. Rental Income)']] #Independent variable
# X = dataset[['University Degree','Profession','Country','Satisfation with employer','Year of Record','Crime Level in the City of Employement','Work Experience in Current Job [years]','Age','Size of City','Body Height [cm]','Yearly Income in addition to Salary (e.g. Rental Income)']] #Independent variable
y = dataset['Total Yearly Income [EUR]']

enc = TargetEncoder(cols=['Gender', 'University Degree','Hair Color','Profession','Country','Satisfation with employer','Housing Situation']).fit(X, y)
# enc = TargetEncoder(cols=['University Degree','Profession','Country','Satisfation with employer','Work Experience in Current Job [years]']).fit(X, y)
ds = enc.transform(X, y)

#Scaling the two features using standard scaler methond
#ss=StandardScaler()
#scaler = ss.fit_transform(ds[['Size of City','Year of Record']])
#scaler=np.transpose(scaler)
#ds['Size of City']=scaler[0]
#ds['Year of Record']=scaler[1]

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.fit(ds)
ds = scaler.transform(ds)


#Splitting the dataset into training(80%) and testing(20%) dataset
X_train, X_test, y_train, y_test = train_test_split(ds, y, test_size=0.1)


#Gradient Boosting Regressor used as a regression algorithm to train the data and predict dependent variable using testing data
#params = {'n_estimators': 1000, 'max_depth': 5, 'min_samples_split': 5,
#          'learning_rate': 0.1, 'loss': 'ls'}
#clf = ensemble.GradientBoostingRegressor(**params)

params = {
          'max_depth': 20,
          'learning_rate': 0.001,
          "boosting": "gbdt",
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1
         }
trn_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_test, label=y_test)
clf = lgb.train(params, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
y_pred = clf.predict(X_test)

#Calculating Root Mean Squared Error
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

#Filling missing values in testing dataset using bfill method
dataset1['Satisfation with employer'] = dataset1['Satisfation with employer'].fillna('Average')

dataset1['Gender'] = dataset1['Gender'].fillna('male')
dataset1['Gender'].replace(['0'], ['missing'], inplace=True)
dataset1['Gender'] = dataset1['Gender'].replace('f','female')

dataset1['Profession'] = dataset1['Profession'].fillna('virtual systems engineer')
dataset1['Profession'] = dataset1['Profession'].str.slice(start=0, stop=3)
dataset1['Profession'] = dataset1['Profession'].str.strip()

dataset1['University Degree'] = dataset1['University Degree'].fillna('Bachelor')
dataset1['University Degree'].replace(['0'], ['missing'], inplace=True)

dataset1['Hair Color'] = dataset1['Hair Color'].fillna('Black')
dataset1['Hair Color'].replace(['0'], ['missing'], inplace=True)

dataset1['Year of Record'] = dataset1['Year of Record'].fillna(1942.0)

dataset1['Yearly Income in addition to Salary (e.g. Rental Income)'] = dataset1['Yearly Income in addition to Salary (e.g. Rental Income)'].replace(regex=['EUR$'], value='')
dataset1['Yearly Income in addition to Salary (e.g. Rental Income)'] = pd.to_numeric(dataset1['Yearly Income in addition to Salary (e.g. Rental Income)'], downcast='float')

dataset1['Work Experience in Current Job [years]'].replace(['#NUM!'], [np.nan], inplace=True)
dataset1['Work Experience in Current Job [years]'] = pd.to_numeric(dataset1['Work Experience in Current Job [years]'])
dataset1['Work Experience in Current Job [years]'].fillna(method="ffill", inplace=True)

dataset1['Housing Situation'].replace(['nA', 0, '0'], ['missing', 'missing1', 'missing2'], inplace=True)
dataset1['Housing Situation'].fillna('missing3', inplace=True)

#Selecting features to predict income for testing data
X = dataset1[['Gender', 'University Degree','Hair Color','Profession','Country','Satisfation with employer','Wears Glasses','Housing Situation','Year of Record','Crime Level in the City of Employement','Work Experience in Current Job [years]','Age','Size of City','Body Height [cm]','Yearly Income in addition to Salary (e.g. Rental Income)']]   #Independent variable
# X = dataset1[['University Degree','Profession','Country','Satisfation with employer','Year of Record','Crime Level in the City of Employement','Work Experience in Current Job [years]','Age','Size of City','Body Height [cm]','Yearly Income in addition to Salary (e.g. Rental Income)']]   #Independent variable

#Target encoding used to encode categorical features of testing dataset using the model used on training dataset
ds1 = enc.transform(X)

ds1 = scaler.transform(ds1)

#Scaling the two features using standard scaler methond
#scaler = ss.fit_transform(ds1[['Size of City','Year of Record']])
#scaler=np.transpose(scaler)
#ds1['Size of City']=scaler[0]
#ds1['Year of Record']=scaler[1]

#Predicting data using Gradient Boosting Regressor
y_pred1 = clf.predict(ds1)

#Writing the predicted income in submission file
dataset2['Total Yearly Income [EUR]'] = y_pred1
dataset2.to_csv('submission.csv',index=False)