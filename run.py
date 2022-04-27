import numpy as np

import prediction

import pickle


genders = ["female", "male"]
outputdir = '/Users/nioushadehestanikolagar/Documents/plot'

hormones = {"female": ["tst", "dhea"],
          "male": ["tst","dhea"]}



# 'growth_spurt', "body-hair", 'skin_change', 'breast_develop', 'menarche',
# 'growth_spurt', "body-hair", 'voice_deep', 'face_hair', 'skin_change',
pds = {
    "female": ['growth_spurt', "body_hair", 'skin_change', 'breast_develop', 'menarche'],
    "male": ['growth_spurt', "body_hair", 'voice_deep', 'face_hair', 'skin_change']
}

for gender in genders:
    filename = '/Users/nioushadehestanikolagar/Documents/ABCD/PhD_project/puberty_measure/data/final_pubertyabcdfinal_{}.csv'.format(gender)
    dataname = 'pubertyabcd_{}'.format(gender)
    predictors = {
        "hormone_age" : hormones[gender],
        "pds_age": pds[gender],
        "pubertyage":hormones[gender] + pds[gender],

    }
    for predictor in predictors:
        print('#'*80)
        print('# Runing: gender=[{}], predictor=[{}]'.format(gender, predictor))
        print('#'*80)
        additional_dataset_test = None

        to_predict = 'age'
        group_by = 'family'
        subject_col = 'ID'

        # group split options
        n_splits = 1
        train_size = .9
        random_state = 0

        #lambdas for gam grid search cross validation
        lams = np.logspace(-5, 5, 50)

        model = prediction.run_predictions(
            filename, f'{dataname}_group_split', predictors[predictor], to_predict, group_by=group_by,
            n_splits=n_splits, train_size=train_size, random_state=random_state,
            split_method='group', runmethod="gam", additional_dataset_test=additional_dataset_test
        )

        outputname = '/Users/nioushadehestanikolagar/Documents/ABCD/PhD_project/puberty_measure/data/final_{}_abcd_{}_model.sav'.format(predictor, gender)
        pickle.dump(model, open(outputname, 'wb'))




