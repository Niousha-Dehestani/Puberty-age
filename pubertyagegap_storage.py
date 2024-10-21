import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GroupKFold

from sklearn.linear_model import LinearRegression

import pickle

import seaborn as sns

import matplotlib.colors as mcolors


# Make a list from gender
genders = ["female", "male"]

# Make a dictionary from hormones for each gender
hormones = {"female": ["tst", "dhea"],
            "male": ["tst","dhea"]}

# Make a dictonary from each sub items of PDS for each gender
pds = {
    "female": ['growth_spurt', "body_hair", 'skin_change', 'breast_develop', 'menarche', Hormones],
    "male": ['growth_spurt', "body_hair", 'voice_deep', 'face_hair', 'skin_change', Hormones]
}

# Define a function to remove regression to the mean
def remove_regression_to_mean_effect(y, y_pred):
    reg = LinearRegression().fit(np.array([y]).T, np.array([y_pred - y]).T)
    rtm_a = reg.coef_[0]
    rtm_b = reg.intercept_
    y_pred_rtm = (y_pred - rtm_b - rtm_a * y)
    return y_pred_rtm

# implemented the fitted model on whole sample with cross validation to calculate the puberty age gap
for gender in genders:
    filename = '/Users/nioushadehestanikolagar/Documents/ABCD/PhD_project/puberty_measure/data/final_pubertyabcdfinal_{}.csv'.format(gender)
    dataname = 'pubertyabcd_{}'.format(gender)
    predictors = {
        "hormone_age": hormones[gender],
        "pds_age": pds[gender],
        "pubertyage": pds[gender] + hormones[gender],

    }
    to_predict_puberty = 'age'
    for predictor in predictors:
        print('#'*80)
        print('# Runing: gender=[{}], predictor=[{}]'.format(gender, predictor))
        print('#'*80)
        additional_dataset_test = None

        group_by = 'family'
        subject_col = 'ID'

        # group split options
        n_splits = 1
        train_size = .9
        random_state = 0

        all_features_puberty = predictors[predictor] + [to_predict_puberty]

        mydfpuberty = pd.read_csv(filename)

        # remove nan
        nanmask = mydfpuberty[all_features_puberty].isna().any(1)
        mycleandf_puberty = mydfpuberty[~nanmask].copy()

        # make sure the features are all float
        mycleandf_puberty[all_features_puberty] = mycleandf_puberty[all_features_puberty].astype(float)

        # prediction features
        Xpuberty = np.array(mycleandf_puberty[predictors[predictor]])

        # standardise the feature scales
        scaler = StandardScaler().fit(Xpuberty)
        Xpuberty = scaler.transform(Xpuberty)
        # Xpuberty = pd.DataFrame(Xpuberty, columns=predictors)

        # variable of interest (to be predicted)
        ypuberty = np.array(mycleandf_puberty[[to_predict_puberty]])

        group_split = np.array(mycleandf_puberty[[group_by]])[:, 0]

        # load fitted model
        loaded_model = pickle.load(
            open(
                '/Users/nioushadehestanikolagar/Documents/ABCD/PhD_project/puberty_measure/data/final_{}_abcd_{}_model.sav'.format(predictor, gender), 'rb'))

        # Out of sample prediction using gam (new code)
        # An inner loop CV = 10
        kfold = GroupKFold(n_splits=10)

        # Calculate the puberty age
        ypuberty_pred = ypuberty[:,0] * 0

        for i, (train_idx, test_idx) in enumerate(kfold.split(X=Xpuberty, y=ypuberty[:,0], groups=group_split)):
            print(f'Split #{i}')

            gamreg = loaded_model.fit(Xpuberty[train_idx,:], ypuberty[train_idx,0])
            ypuberty_pred[test_idx] = gamreg.predict(Xpuberty[test_idx, :])

        print('Model fitted...')

        ypuberty_gap = ypuberty_pred - ypuberty[:,0]
        mycleandf_puberty['{}_abcd'.format(predictor)] = ypuberty_pred
        mycleandf_puberty['{}_gap_abcd'.format(predictor)] = ypuberty_gap

        mycleandf_puberty["{}_abcd_rtm".format(predictor)] = remove_regression_to_mean_effect(
        mycleandf_puberty['age'],
        mycleandf_puberty["{}_abcd".format(predictor)]
            )

        mycleandf_puberty["{}_abcd_gap_rtm".format(predictor)] = \
        mycleandf_puberty["{}_abcd_rtm".format(predictor)] - mycleandf_puberty["age"]



        mycleandf_puberty.to_csv(
            r'/Users/nioushadehestanikolagar/Documents/ABCD/PhD_project/puberty_measure/data/mycleandf_{}_gap_abcd_{}.csv'.format(predictor, gender), index=True,
            header=True)

        mycleandf_puberty = mycleandf_puberty.set_index("ID-wave")


        # remove nan
        # print(mydfpuberty.columns)
        nanmask = mycleandf_puberty[["pds", "age"]].isna().any(1)
        mycleandf_puberty = mycleandf_puberty[~nanmask].copy()

        # make sure the features are all float
        mycleandf_puberty[["pds", "age"]] = mycleandf_puberty[["pds", "age"]].astype(float)

        x = mycleandf_puberty["age"]
        y = mycleandf_puberty["pds"]
        reg = LinearRegression().fit(np.array([x]).T, np.array([y]).T)
        reg_a = reg.coef_[0]
        reg_b = reg.intercept_
        y_reg_x = (y - reg_b - reg_a * x)

        mycleandf_puberty["pds_age_regressed"] = y_reg_x
        mycleandf_puberty["pds_age_regress_gap_abcd"] = y_reg_x
        mycleandf_puberty["pds_age_regress_abcd"] = y_reg_x
        mycleandf_puberty["pds_age_regress_abcd_rtm"] = y_reg_x
        mycleandf_puberty["pds_age_regress_abcd_gap_rtm"] = y_reg_x


        # mycleandf_puberty = mycleandf_puberty.set_index("ID-wave")

        mycleandf_puberty.to_csv(
            r'directory/mycleandf_pds_age_regressgap_without_{}.csv'.format(
                gender),
            index=True, header=True)

