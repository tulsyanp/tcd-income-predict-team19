import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.externals import joblib
import matplotlib.pyplot as plt


def model_linear_regression_predict(X, y, predictionDataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)

    y_pred = linear_regression.predict(X_test)

    calculate_score(y_test, y_pred)

    predict(linear_regression, predictionDataFrame)


def model_random_forest_predict(X, y, predictionDataFrame):
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=10)

    random_forest = RandomForestRegressor(n_estimators=50, random_state=10)
    random_forest.fit(train_features, train_labels)

    predictions = random_forest.predict(test_features)

    calculate_score(test_labels, predictions)

    predict(random_forest, predictionDataFrame)


def model_xgboost_predict(X, y, predictionDataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    xgb = XGBClassifier(max_depth=20, learning_rate=0.001)
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)

    calculate_score(y_test, y_pred)

    predict(xgb, predictionDataFrame)


def model_lightbgm_predict(X, y, predictionDataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    params = {
        'max_depth': 20,
        'learning_rate': 0.001,
        "boosting": "gbdt",
        "bagging_seed": 10,
        "metric": 'mae',
        "verbosity": -1,
    }

    train_data = lgb.Dataset(X, label=y)
    test_data = lgb.Dataset(X_test, label=y_test)

    model = lgb.train(params, train_data, 200000, valid_sets=[train_data, test_data], verbose_eval=1000, early_stopping_rounds=500)

    saved_model = joblib.dump(model, 'lgb_model1.pkl')

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    calculate_score(y_test, y_pred)

    generate_graph(y_test, y_pred)

    predict(model, predictionDataFrame)


def calculate_score(test, pred):
    print('Internal Test: Variance score: %.2f' % r2_score(test, pred))
    print("Internal Test: Root Mean squared error: %.2f" % math.sqrt(mean_squared_error(test, pred)))

    errors = abs(pred - test)
    mape = 100 * (errors / test)
    accuracy = 100 - np.mean(mape)

    print('Internal Test: Mean Absolute Error:', round(np.mean(errors), 2))
    # print('Internal Test: Accuracy:', accuracy_score(test, pred))
    print(accuracy)


def generate_csv(prediction):
    submission = pd.read_csv("submission-sample.csv")
    submission['Total Yearly Income [EUR]'] = prediction
    submission.to_csv('submission.csv', index=False)

    return submission


def predict(model, predictionDataFrame):
    # P_test = predictionDataFrame.loc[:, predictionDataFrame.columns != 'total_income']
    P_pred = np.exp(model.predict(predictionDataFrame, num_iteration=model.best_iteration))
    generate_csv(P_pred)
    print('Predicted CSV generated')


def generate_graph(test, pred):
    df = pd.DataFrame({'Actual': test, 'Predicted': pred})
    df1 = df.head(50)
    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig('Actual Vs. Prediction.png')
