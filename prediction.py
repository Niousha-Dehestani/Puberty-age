import os
import pandas as pd
from pygam.datasets import wage
from pygam import LinearGAM, s, f
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines, CyclicCubicSplines
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
import pickle
# from statsmodels.gam.tests.test_penalized import df_autos

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
    print("-"*80)
    print(f"{title}\n")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred)}")
    print(f"Pearson's correlation: {np.corrcoef(y_true, y_pred)[0, 1]}")
    print(f'correlation: {stats.pearsonr(y_true, y_pred)}')
    print("-"*80)


# Define a function to report the t statistic for prediction model
def report_prediction_statistic(y_pred1, y_pred2, y_true, title = 'T_test performance'):
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


def split_by_group_limited(x, y, df, group_by, subject_col, train_count=None, n_splits=1, train_size=0.7, random_state=0):
    # split test & train such that only one wave of every subject is used in training
    if train_count is None:
        train_count = df[subject_col].nunique()
    subject_count = df[subject_col].nunique()
    train_size_limit = min(train_count, (train_size * subject_count))
    test_size_limit = max(((1 - train_size) * subject_count), (df[subject_col].nunique() - train_count))
    np.random.seed(random_state)
    indices = [x for x in df.index]
    np.random.shuffle(indices)
    train_subjects = set()
    train_idx = []
    test_idx = []

    for idx in indices:
        subject_id = df.iloc[idx][group_by]
        if len(train_idx) < train_size_limit:
            # check to add to train
            train_subjects.add(subject_id)
            train_idx.append(idx)
        elif len(test_idx) < test_size_limit:
            # check to add to test
            if subject_id not in train_subjects:
                test_idx.append(idx)

    x_train, x_test, y_train, y_test = (
       np.array(x.iloc[train_idx]),
       np.array(x.iloc[test_idx]),
       np.array(y.iloc[train_idx])[:,0],
       np.array(y.iloc[test_idx])[:,0]
    )
    return x_train, x_test, y_train, y_test


def split_by_subject_and_wave(x, y, df, subject_col, train_count=None, n_splits=1, train_size=0.7, random_state=0):
    # split test & train such that only one wave of every subject is used in training
    if train_count is None:
        train_count = df[subject_col].nunique()
    subject_count = df[subject_col].nunique()
    train_size_limit = min(train_count, (train_size * subject_count))
    np.random.seed(random_state)
    indices = [x for x in df.index]
    np.random.shuffle(indices)
    train_subjects = set()
    train_idx = []
    test_idx = []

    for idx in indices:
        subject_id = df.iloc[idx][subject_col]
        if len(train_idx) < train_size_limit:
            # check to add to train
            if subject_id not in train_subjects:
                train_subjects.add(subject_id)
                train_idx.append(idx)
        else:
            # check to add to test
            if subject_id not in train_subjects:
                test_idx.append(idx)

    x_train, x_test, y_train, y_test = (
        np.array(x.iloc[train_idx]),
        np.array(x.iloc[test_idx]),
        np.array(y.iloc[train_idx])[:,0],
        np.array(y.iloc[test_idx])[:,0]
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
    nanmask = mydf[all_features].isna().any(axis=1)
    mycleandf = mydf[~nanmask].copy()

    # make sure the features are all float
    mycleandf[all_features] = mycleandf[all_features].astype(float)

    # prediction features
    x = mycleandf[predictors]

    # standardise the features
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    x = pd.DataFrame(x, columns=predictors)
    outputname_scaler = f'/home/newsha/Documents/myJupyter/paper3/double_check/data_finalise/scaler_{dataname}_harm_gam.sav'

    pickle.dump(scaler, open(outputname_scaler, 'wb'))

    # variable of interest (to be predicted)
    y = mycleandf[[to_predict]]

    if split_method == 'group':
        # split test & train with a group-wise split
        if group_by is None:
            raise Exception(f"Group_by not provided")
        x_train, x_test, y_train, y_test = split_by_group(x, y, mycleandf, group_by, n_splits=n_splits, train_size=train_size, random_state=random_state)

    elif split_method == 'group-limit-by-subject':
        # split test & train with a group-wise split
        if group_by is None:
            raise Exception(f"Group_by not provided")
        if subject_col is None:
            raise Exception(f"Subject_col not provided")
        x_train, x_test, y_train, y_test = split_by_group_limited(x, y, mycleandf, group_by, subject_col, n_splits=n_splits, train_size=train_size, random_state=random_state)

    elif split_method == 'subject':
        if subject_col is None:
            raise Exception(f"Subject_col not provided")
        x_train, x_test, y_train, y_test = split_by_subject_and_wave(x, y, mycleandf, subject_col, n_splits=n_splits, train_size=train_size, random_state=random_state)

    else:
        raise Exception(f"Invalid split method: {split_method}")


    # choose the method for run
    model = None

    if runmethod == "gam":
        gamreg = LinearGAM()
        gamreg = gamreg.gridsearch(x_train, y_train, lam=lams, objective='GCV')
        gamfit = gamreg.fit(x_train, y_train)
        y_pred = gamfit.predict(x_test)
        yt_pred = gamfit.predict(x_train)
        model = gamfit
        # parameters = [{'lam': lams}]
        # gamgrid = GridSearchCV(gamreg, parameters, scoring="r2", n_jobs=-1, cv=10)
        # gamgrid_fit = gamgrid.fit(x_train, y_train)
        # y_pred = gamgrid_fit.predict(x_test)
        # yt_pred = gamgrid_fit.predict(x_train)
        # model = gamgrid_fit

        # # statsmodel
        # # tmpdf = pd.DataFrame(x_train, columns=predictors)
        # # tmpdf[to_predict] = y_train
        # # x_spline = tmpdf[predictors]
        # bs = BSplines(x_train, df=10*np.ones(len(predictors)).astype(int), degree=3*np.ones(len(predictors)).astype(int))
        # bs = CyclicCubicSplines(x_train, df=10*np.ones(len(predictors)).astype(int))
        #
        # alpha = 1000*np.ones(len(predictors))
        # # terms = ' + '.join(predictors)
        # # gam_bs = GLMGam.from_formula(f'{to_predict} ~ {terms}', data=tmpdf, smoother=bs, alpha=alpha)
        # gam_bs = GLMGam(y_train, smoother=bs, alpha=alpha)
        # gam_bs_fit = gam_bs.fit()
        # print(gam_bs_fit.summary())
        # # y_pred = gam_bs_fit.predict(BSplines(x_train, df=10 * np.ones(len(predictors)).astype(int), degree=3 * np.ones(len(predictors)))
        #
        # yt_pred = gam_bs_fit.predict(bs.transform(x_train))
        # y_pred = gam_bs_fit.predict(bs.transform(x_test))
        # model = gam_bs_fit

    elif runmethod == "svrrbf":
        svr_rbf = svm.SVR()
        parameters = [{'kernel': ['rbf'], 'gamma': svrrbf_gammas, 'C': c_svrrbf, 'epsilon': epsilon_svrrbf}]
        svrgrid = GridSearchCV(svr_rbf, parameters, scoring="r2", n_jobs=-1, cv=cv_svrrbf,)
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
            scores = ['MAE', 'RMSE', 'corr', 'EV']
            for score in scores:
                score_mean = np.mean([x[score] for x in performances])
                print(f"report_crossvalidation_scores : {(score_mean)}")


    elif runmethod == "linearregression":
        linearreg = LinearRegression()
        linearregfit = linearreg.fit(x_train, y_train)
        y_pred = linearregfit.predict(x_test)
        yt_pred = linearregfit.predict(x_train)
        model = linearregfit


        
    elif runmethod == "elastic":

        # Create an ElasticNet object
        elastic_net = ElasticNetCV(cv=5, random_state=0, l1_ratio =[.1, .5, .7, .9, .95, .99, 0.999, 1])

        # Fit the model to the training data
        elastic_net_fit = elastic_net.fit(x_train, y_train)
        # Make predictions on the test data
        y_pred = elastic_net_fit.predict(x_test)
        yt_pred = elastic_net_fit.predict(x_train)
        model = elastic_net_fit
        
    elif runmethod == "lasso":
        lasso_cv = LassoCV(cv=5)
        # Fit the model to the training data
        lassofit = lasso_cv.fit(x_train, y_train)
        # Make predictions on the test data
        y_pred = lassofit.predict(x_test)
        yt_pred = lassofit.predict(x_train)
        model = lassofit

        
    elif runmethod == "ridge":
        ridge_cv = RidgeCV(cv=5)
        ridgefit = ridge_cv.fit(x_train, y_train)
        y_pred = ridgefit.predict(x_test)
        yt_pred = ridgefit.predict(x_train)
        model = ridgefit  


    # report accuracy of the model performance before removing regression to the mean
    report_prediction_accuracy(y_train, yt_pred, title='Prediction performance (train): combined')
    report_prediction_accuracy(y_test, y_pred, title='Prediction performance (test): combined')

    # remove regression to mean effect:
    yt_pred_rtm, y_pred_rtm = remove_regression_to_mean_effect(y_train, yt_pred, y_test, y_pred)

    # report accuracy of the model performance after removing regression to the mean
    report_prediction_accuracy(y_test, y_pred_rtm, title='Prediction performance (after RTM, not valid): combined')


    # Rerun predictions with single predictors
    if testindividual_predictors:
        for pidx, predictor in enumerate(predictors):
            # LinearGAM defaults to splines for every single feature
            gamreg_single = LinearGAM()
            gamreg_single = gamreg_single.gridsearch(x_train[:,pidx][:,np.newaxis], y_train, lam=lams)
            gamfit_single = gamreg_single.fit(x_train[:,pidx][:,np.newaxis], y_train)

            # predict using fitted model
            y_pred_single = gamreg_single.predict(x_test[:,pidx][:,np.newaxis])
            yt_pred_single = gamreg_single.predict(x_train[:,pidx][:,np.newaxis])

            # report accuracy before RTM
            report_prediction_accuracy(y_train, yt_pred_single, title=f'Prediction performance (train): single, {predictor}')
            report_prediction_accuracy(y_test, y_pred_single, title=f'Prediction performance (test): single, {predictor}')

            # report t statistic
            report_prediction_statistic(y_pred, y_pred_single, y_test, title = f'T_test(test): single, {predictor}')


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

        #accuracy
        additionaly = additionaly.values.flatten()
        report_prediction_accuracy(additionaly, additionaly_pred, title='Prediction performance additional dataset')
        plot_prediction_performance(y_train, yt_pred, additionaly, additionaly_pred, dataname=f'additional-{dataname}',
                                   outputdir=outputdir,
                                   y_label='age', pred_label='predicted_age')

    return model
    # return [additionaly_pred, additionaly]

