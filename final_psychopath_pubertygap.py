import statsmodels.api as sm

import pandas

import statsmodels.api as sm

from statsmodels.stats.mediation import Mediation

import numpy as np


import pandas as pd

import statsmodels.formula as smf

from statsmodels.formula.api import ols

from statsmodels.stats.api import anova_lm

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

import statsmodels.formula.api as smf

import pickle

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score

import pickle

from pymer4.models import Lmer

import pandas as pd

from statsmodels.stats.multitest import fdrcorrection

from sklearn.preprocessing import quantile_transform


# Define a function for implemting mixed effect model
def run_lmer(formula, data, summarize=False):
    model = Lmer(
        formula,
        data=data
    )
    fitted_model = model.fit(
        summarize=summarize,
    )
    if summarize:
        print(fitted_model)

    return model


# Read file for data collection sites
abcd_site = pd.read_csv("/Users/nioushadehestanikolagar/Documents/ABCD/Enviornment/multisite.csv")

# Rename the columns names
abcd_site = abcd_site.rename(columns={

    'SUBJECTKEY': 'ID',
    'EVENTNAME': 'eventname',
    'SITE_ID_L': 'multisite',

})

# Creat the column with mix of wave and ID of each person
abcd_site['ID-wave'] = abcd_site.ID + '-' + abcd_site.eventname

# Convert each site from string to number
abcd_site["multisite"] = abcd_site["multisite"].replace({
    'site01': 1,
    'site02': 2,
    'site03': 3,
    'site04': 4,
    'site05': 5,
    'site06': 6,
    'site07': 7,
    'site08': 8,
    'site09': 9,
    'site10': 10,
    'site11': 11,
    'site12': 12,
    'site13': 13,
    'site14': 14,
    'site15': 15,
    'site16': 16,
    'site17': 17,
    'site18': 18,
    'site19': 19,
    'site20': 20,
    'site21': 21,
    "site22": 22,
})

# Index the "ID-wave" column
abcd_site = abcd_site.set_index("ID-wave")

abcd_site = abcd_site[['multisite']]

# Read the psychopathogy dimentions
abcdcbcl = pd.read_csv('/Users/nioushadehestanikolagar/Documents/ABCD/PhD_project/puberty_measure/data/cbclnew.csv', low_memory=False)

# Removing the header from dataframe
abcdcbcl = abcdcbcl[1:]

# Choose the psychopathology dimentions from CBCL questionairs


# Rename cbcl columns
abcdcbcl = abcdcbcl.rename(columns={
    'SUBJECTKEY': 'ID',
    "EVENTNAME" : "eventname",
    "CBCL_SCR_SYN_INTERNAL_T":'Internalising',
    "CBCL_SCR_SYN_EXTERNAL_T": 'Externalising',
    "CBCL_SCR_DSM5_DEPRESS_T" : 'Depression',
    "CBCL_SCR_DSM5_ANXDISORD_T": 'Anxiety',
    "CBCL_SCR_DSM5_SOMATICPR_T": 'Somatic',
    "CBCL_SCR_DSM5_ADHD_T": 'ADHD',
    "CBCL_SCR_DSM5_OPPOSIT_T": 'Opposite',
    "CBCL_SCR_DSM5_CONDUCT_T": 'Conduct',
    "CBCL_SCR_07_SCT_T": 'SCT',
    "CBCL_SCR_07_OCD_T": 'OCD',
    "CBCL_SCR_07_STRESS_T" : 'Stress',
    "CBCL_SCR_SYN_TOTPROB_T": 'Total',
})

psychopathologies = [
    "Internalising", "Externalising", "Depression",
    "Anxiety", "Somatic", "ADHD",
    "Opposite","Conduct", "SCT", "OCD",
    "Stress", "Total"
]


abcdcbcl = abcdcbcl[['ID', 'eventname'] + psychopathologies]


# Create the columns called 'ID-wave' from wave and ID for each person
abcdcbcl['ID-wave'] = abcdcbcl.ID + '-' + abcdcbcl.eventname

# Index 'ID-wave'
abcdcbcl = abcdcbcl.set_index('ID-wave')

# Removing the nan from these columns
nanmask = abcdcbcl.isna().any(1)
abcdcbcl = abcdcbcl[~nanmask].copy()

# Make sure the features are all float
abcdcbcl[psychopathologies] = abcdcbcl[psychopathologies].astype(float)

abcdcbcl = abcdcbcl[psychopathologies]

# Implemented Mixed effect model for each psychopathology
datadict = {}
extradatadict = {}
extraindex = 0
for psychopathology in psychopathologies:
    datadict[psychopathology] = {}

genders = ["female", "male"]

hormones = {"female": ["tst", "dhea"],
            "male": ["tst","dhea"]}

pds = {
    "female": ['growth_spurt', "body-hair", 'skin_change', 'breast_develop', 'menarche'],
    "male": ['growth_spurt', "body-hair", 'voice_deep', 'face_hair', 'skin_change']
}


for gender in genders:
    predictors = {
        "pds_age": pds[gender],
        # "pds_age_regress": None,
        "pubertyage": pds[gender] + hormones[gender],
        # "pubertyage_int": hormones[gender] + pds[gender] + get_interactions(pds[gender] + hormones[gender] ),
        "hormone_age" : hormones[gender],
        # "traditional": ["pds"]
    }

    for predictor in predictors:
        print('#'*80)
        print('# Runing: gender=[{}], predictor=[{}]'.format(gender, predictor))
        print('#'*80)

        # Read each puberty age model's file
        puberty_age = pd.read_csv('/Users/nioushadehestanikolagar/Documents/ABCD/PhD_project/puberty_measure/data/mycleandf_{}_gap_abcd_{}.csv'. format(predictor, gender))

        # Index 'ID-wave' column
        puberty_age = puberty_age.set_index('ID-wave')

        puberty_age = puberty_age[["{}_abcd_gap_rtm".format(predictor), "ID", "age", "eventname", "family", "race"]]

        # Conctat psychopathology file with
        puberty_cbcl = pd.concat([abcdcbcl, puberty_age], axis=1)

        # Duplicate the columns
        puberty_cbcl.drop(['ID', 'eventname', ], axis=1, inplace=True)
        puberty_cbcl['ID-wave-copy'] = puberty_cbcl.index.copy()
        puberty_cbcl[['ID', 'eventname']] = puberty_cbcl[
            'ID-wave-copy'].str.split('-', expand=True)

        # Remove nan from columns
        puberty_cbcl = puberty_cbcl[
            ~(

                    puberty_cbcl["{}_abcd_gap_rtm".format(predictor)].isna() |
                    puberty_cbcl["Internalising"].isna() |
                    puberty_cbcl['family'].isna()

            )].copy()

        # Standarize features
        features_to_standardize = ['age','{}_abcd_gap_rtm'.format(predictor)]

        standardized_feature_names = ['{}_std'.format(x) for x in features_to_standardize]

        puberty_cbcl[standardized_feature_names] = StandardScaler().fit(
        puberty_cbcl[features_to_standardize]).transform(
        puberty_cbcl[features_to_standardize])

        # Concat site with puberty and cbcl
        puberty_cbcl = pd.concat([abcd_site, puberty_cbcl], axis=1)
        puberty_cbcl = puberty_cbcl[
        ~(puberty_cbcl[["multisite", "family", "age"]].isna().any(1))
        ].copy()

        # Choose one sibiling from each family randomly
        selected_IDs = list(
            puberty_cbcl[['family', 'ID-wave-copy']].groupby('family', group_keys=False).apply(
                lambda df: df.sample(1, random_state=1))['ID-wave-copy'])
        puberty_cbcl_unrelated = puberty_cbcl[
            puberty_cbcl['ID-wave-copy'].isin(selected_IDs)].copy()


        # Convert variable to categorise variables
        puberty_cbcl_unrelated.multisite = puberty_cbcl_unrelated.multisite.astype(
            str).astype("category")
        puberty_cbcl_unrelated.family = puberty_cbcl_unrelated.family.astype(
            str).astype("category")
        puberty_cbcl_unrelated.ID = puberty_cbcl_unrelated.ID.astype(
            str).astype("category")

        print('# Dataframe shape: {}'.format(puberty_cbcl_unrelated.shape))


        # Implement the mixed effect model
        for psychopathology in psychopathologies:
            print('# Psychopathology: {}'.format(psychopathology))
            modelabcd = run_lmer(
                '{} ~ {}_abcd_gap_rtm_std + age_std + (1|multisite)'.format(psychopathology, predictor),
                puberty_cbcl_unrelated,
            )

            datadict[psychopathology]['{}_abcd_gap_rtm_std-{}-estimate'.format(predictor, gender)] = modelabcd.coefs['Estimate']["{}_abcd_gap_rtm_std".format(predictor)]
            datadict[psychopathology]['{}_abcd_gap_rtm_std-{}-pval'.format(predictor, gender)] = modelabcd.coefs['P-val']["{}_abcd_gap_rtm_std".format(predictor)]
            datadict[psychopathology]['{}_abcd_gap_rtm_std-{}-tstat'.format(predictor, gender)] = modelabcd.coefs['T-stat']["{}_abcd_gap_rtm_std".format(predictor)]
            datadict[psychopathology]['{}_abcd_gap_rtm_std-{}-aic'.format(predictor, gender)] = modelabcd.AIC

            extradatadict[extraindex] = {}
            extradatadict[extraindex]['predictor'] = predictor
            extradatadict[extraindex]['gender'] = gender
            extradatadict[extraindex]['psychopathology'] = psychopathology
            extradatadict[extraindex]['tstat'] = modelabcd.coefs['T-stat']["{}_abcd_gap_rtm_std".format(predictor)]
            extraindex += 1


# Save t-statistical, P value and AIC of the each puberty age model
finaldf = pd.DataFrame(datadict).transpose()

finaldf.to_csv(
    r'/Users/nioushadehestanikolagar/Documents/ABCD/PhD_project/puberty_measure/data/psychopath_puberty2.csv',
    index=True, header=True
)

extradf = pd.DataFrame(extradatadict).transpose()

extradf.to_csv(
    r'/Users/nioushadehestanikolagar/Documents/ABCD/PhD_project/puberty_measure/data/psychopath_puberty_extra.csv',
    index=True, header=True
)





