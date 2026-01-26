"""
Script: 02_run_models.py
Description: Loops through male and female data to train Puberty GAM models 
             using ComBat-harmonized hormones and PDS to predict chronological age.
             Saves the final trained models AND scalers to the processed folder.
"""

import numpy as np
import pickle
from pathlib import Path
import prediction 

# --- PATHS ---
# UPDATE THIS PATH to your local ABCD data directory before running.
# Data is kept separate from code for privacy and storage constraints.
DATA_DIR = Path('../../data/processed')

# Using the ComBat harmonized dataset
INPUT_FILE = DATA_DIR / '01b_harmonized_data.csv'

# Define genders to loop through (1 = Male, 2 = Female)
genders = ["1", "2"]

# --- DEFINE ABCD 6.0 COLUMN NAMES ---
# Residualized Hormones
hormones = {
    "2": ['tst_filter_rsi_hormone', 'dhea_filter_rsi_hormone'],
    "1": ['tst_filter_rsi_hormone', 'dhea_filter_rsi_hormone']
}

# PDS Items (ABCD uses 001, 002, 003 for both sexes to cover growth/hair/skin/voice/menarche)
pds = {
    "2": ['ph_p_pds_001', 'ph_p_pds_002', 'ph_p_pds_003', 'ph_p_pds__f_002', 'ph_p_pds__f_001'],
    "1": ['ph_p_pds_001', 'ph_p_pds_002', 'ph_p_pds_003', 'ph_p_pds__m_001','ph_p_pds__m_002']
}

# ----------------------------------------------------
# MAIN EXECUTION LOOP
# ----------------------------------------------------
if __name__ == "__main__":
    for gender in genders:
        # Define the three predictor models to test
        predictors = {
            "hormone_only": hormones[gender],
            "pds_only": pds[gender],
            "combined": hormones[gender] + pds[gender]
        }
        
        for pred_name, pred_vars in predictors.items():
            print('#' * 80)
            print(f'# Running: Gender=[{gender}], Model=[{pred_name}]')
            print('#' * 80)

            # ABCD 6.0 specific columns for age and family grouping
            to_predict = 'ab_g_dyn__visit_age'
            group_by = 'ab_g_stc__design_id__fam'

            # Modeling parameters
            n_splits = 1
            train_size = 0.9
            random_state = 0
            
            # Non-linear tuning parameters for GAM
            lams = np.logspace(-5, 5, 50)

            # --- RUN THE MODEL ---
            # NOTE: We catch BOTH the model and the scaler to prevent data leakage in future tests
            model, scaler = prediction.run_predictions(
                filename=INPUT_FILE, 
                dataname=f'puberty_abcd_gender{gender}', 
                predictors=pred_vars, 
                to_predict=to_predict, 
                group_by=group_by,
                split_method='group',  
                n_splits=n_splits, 
                train_size=train_size, 
                random_state=random_state,
                lams=lams
            )

            # --- SAVE THE MODEL ---
            model_output = DATA_DIR / f'final_{pred_name}_abcd_{gender}_model.sav'
            with open(model_output, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to: {model_output}")

            # --- SAVE THE SCALER ---
            scaler_output = DATA_DIR / f'final_{pred_name}_abcd_{gender}_scaler.pkl'
            with open(scaler_output, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Scaler saved to: {scaler_output}\n")
