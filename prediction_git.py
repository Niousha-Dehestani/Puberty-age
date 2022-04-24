import os

import pandas as pd

import numpy as np

from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

from sklearn import svm

from sklearn.model_selection import GridSearchCV

from sklearn.dummy import DummyRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GroupShuffleSplit

from sklearn.linear_model import LinearRegression

from scipy import stats

from sklearn.model_selection import KFold, cross_val_score


# Define a function to remove regression to the mean
def remove_regression_to_mean_effect(y_train, y_train_pred, y_test, y_test_pred):
    reg = LinearRegression().fit(np.array([y_train]).T, np.array([y_train_pred - y_train]).T)
    rtm_a = reg.coef_[0]
    rtm_b = reg.intercept_
    y_train_pred_rtm = (y_train_pred - rtm_b - rtm_a * y_train)
    y_test_pred_rtm = (y_test_pred - rtm_b - rtm_a * y_test)
    return y_train_pred_rtm, y_test_pred_rtm


# Define a function to calculate the accuracy of the model
def report_prediction_accuracy(y_true, y_pred, title='Prediction performance:'):
    print("-" * 80)
    print(f"{title}\n")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred)}")
    print(f"Pearson's correlation: {np.corrcoef(y_true, y_pred)[0, 1]}")
    print(f'correlation: {stats.pearsonr(y_true, y_pred)}')
    print("-" * 80)


# Define a function to report the t statistic for prediction model
def report_prediction_statistic(y_pred1, y_pred2, y_true, title='T_test performance'):
    print("-" * 80)
    print(f"{title}\n")
    y_pred1_gap = abs(y_pred1 - y_true)
    y_pred2_gap = abs(y_pred2 - y_true)
    print(f"t-test results: {stats.ttest_ind(y_pred1_gap, y_pred2_gap, equal_var=False)}")
    print("-" * 80)


# Compute prediction performance metrics
def get_prediction_performance(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'corr': np.corrcoef(y_true, y_pred)[0, 1],
    }


# Split dataset into train and test
def split_by_group(x, y, df, group_by, n_splits=1, train_size=0.7, random_state=0):
    # split test & train with a group-wise split
    gss = GroupShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(x, y, df[group_by]))
    x_train, x_test, y_train, y_test = (
        np.array(x.iloc[train_idx]),
        np.array(x.iloc[test_idx]),
        np.array(y.iloc[train_idx])[:, 0],
        np.array(y.iloc[test_idx])[:, 0]
    )
    return x_train, x_test, y_train, y_test


# Define a function to predict chronological age from multiple features
def run_predictions(filename, dataname, predictors, to_predict, group_by=None, subject_col=None,
                    runmethod="gam", n_splits=1, train_size=0.7, random_state=0, testindividual_predictors=False,
                    lams=np.logspace(-4, 4, 40), split_method=None, svrrbf_gammas=[1e-4, 1e-3, 0.01, 0.1, 1],
                    c_svrrbf=[0.1, 1, 10, 100, 1000], epsilon_svrrbf=[0.1, 0.3, 0.5, 0.7, 0.9], cv_svrrbf=5,
                    additional_dataset_test=None, report_crossvalidation_scores=None,
                    ):
    # initializations
    all_features = predictors + [to_predict]

    # read file
    mydf = pd.read_csv(filename)

    # remove nans
    print(mydf.columns)
    nanmask = mydf[all_features].isna().any(1)
    mycleandf = mydf[~nanmask].copy()

    # make sure the features are all float
    mycleandf[all_features] = mycleandf[all_features].astype(float)

    # prediction features
    x = mycleandf[predictors]

    # standardise the features
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    x = pd.DataFrame(x, columns=predictors)

    # variable of interest (to be predicted)
    y = mycleandf[[to_predict]]

    if split_method == 'group':
        # split test & train with a group-wise split
        if group_by is None:
            raise Exception(f"Group_by not provided")
        x_train, x_test, y_train, y_test = split_by_group(x, y, mycleandf, group_by, n_splits=n_splits,
                                                          train_size=train_size, random_state=random_state)

    else:
        raise Exception(f"Invalid split method: {split_method}")

    # Use SVR to predict chronological age
    model = None

    if runmethod == "svrrbf":
        svr_rbf = svm.SVR()
        parameters = [{'kernel': ['rbf'], 'gamma': svrrbf_gammas, 'C': c_svrrbf, 'epsilon': epsilon_svrrbf}]
        svrgrid = GridSearchCV(svr_rbf, parameters, scoring="r2", n_jobs=-1, cv=cv_svrrbf, )
        svr_rbffit = svrgrid.fit(x_train, y_train)
        y_pred = svr_rbffit.predict(x_test)
        yt_pred = svr_rbffit.predict(x_train)
        model = svr_rbffit
        if report_crossvalidation_scores is not None:
            cv_C = model.best_params_['C']
            cv_epsilon = model.best_params_['epsilon']
            cv_gamma = model.best_params_['gamma']
            k_fold = KFold(n_splits=5)
            performances = []
            for learn_indices, validation_indices in k_fold.split(x_train):
                x_learn, x_validate = x_train[learn_indices], x_train[validation_indices]
                y_learn, y_validate = y_train[learn_indices], y_train[validation_indices]
                svr_val = svm.SVR(C=cv_C, epsilon=cv_epsilon, gamma=cv_gamma)
                svr_val.fit(x_learn, y_learn)
                y_validate_pred = svr_val.predict(x_validate)
                performance = get_prediction_performance(y_validate, y_validate_pred)
                performances.append(performance)
            scores = ['MAE', 'RMSE', 'corr', 'EV']
            for score in scores:
                score_mean = np.mean([x[score] for x in performances])
                print(f"report_crossvalidation_scores : {(score_mean)}")

    # report accuracy of the model performance before removing regression to the mean
    report_prediction_accuracy(y_train, yt_pred, title='Prediction performance (train): combined')
    report_prediction_accuracy(y_test, y_pred, title='Prediction performance (test): combined')

    # remove regression to mean effect:
    yt_pred_rtm, y_pred_rtm = remove_regression_to_mean_effect(y_train, yt_pred, y_test, y_pred)

    # report accuracy of the model performance after removing regression to the mean
    report_prediction_accuracy(y_test, y_pred_rtm, title='Prediction performance (after RTM, not valid): combined')

    # remove regression to mean effect:
    yt_pred_rtm, y_pred_rtm = remove_regression_to_mean_effect(y_train, yt_pred, y_test, y_pred)

    # report accuracy after RTM
    report_prediction_accuracy(y_test, y_pred_rtm, title='Prediction performance (after RTM, not valid): combined')

    # you can use this fitted model in your dataset by changing additional_dataset_test to a csv file with approapriate format
    if additional_dataset_test is not None:
        # readfile
        additionaldf = pd.read_csv(additional_dataset_test)

        # remove nan
        additionalnanmask = additionaldf[all_features].isna().any(1)
        additionalcleandf = additionaldf[~additionalnanmask].copy()

        # make sure the features are all float
        additionalcleandf[all_features] = additionalcleandf[all_features].astype(float)

        # prediction features
        additionalx = additionalcleandf[predictors]

        # standardise the feature scales
        additionalx = scaler.transform(additionalx)
        additionalx = pd.DataFrame(additionalx, columns=predictors)

        # variable of interest (to be predicted)
        additionaly = additionalcleandf[[to_predict]]

        additionaly_pred = model.predict(additionalx)

        # accuracy
        additionaly = additionaly.values.flatten()
        report_prediction_accuracy(additionaly, additionaly_pred, title='Prediction performance additional dataset')

    return model
    # return [additionaly_pred, additionaly]
