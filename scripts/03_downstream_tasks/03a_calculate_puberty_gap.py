"""
Script: 03a_calculate_puberty_gap.py
Description: 
    1. Loads harmonized data.
    2. Runs 10-fold Cross-Validation to generate UNBIASED predictions.
    3. Calculates the Puberty Gap (Predicted - Actual).
    4. Applies Regression-to-the-Mean (RTM) correction.
    5. Saves the final clinical dataset to your processed folder.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from pathlib import Path

# --- PATHS (Your Local Mac Setup) ---
DATA_DIR = Path('/Users/nioushad/Documents/abcd6.0/processed')
INPUT_FILE = DATA_DIR / '01b_harmonized_data.csv'
OUTPUT_FILE = DATA_DIR / '03_final_dataset_with_pag_rtm.csv'

# Define Genders (1=Male, 2=Female)
genders = ["1", "2"] 

# Define Features (Must match exactly what was used in Step 02)
hormones_list = ['tst_filter_rsi_hormone', 'dhea_filter_rsi_hormone']

# Sex-specific PDS items
pds_dict = {
    "2": ['ph_p_pds_001', 'ph_p_pds_002', 'ph_p_pds_003', 'ph_p_pds__f_002', 'ph_p_pds__f_001'],
    "1": ['ph_p_pds_001', 'ph_p_pds_002', 'ph_p_pds_003', 'ph_p_pds__m_001', 'ph_p_pds__m_002']
}

# --- HELPER FUNCTION: RTM CORRECTION ---
def remove_regression_to_mean_effect(y_true, y_pred):
    """
    Corrects the bias where models overpredict young subjects and underpredict older ones.
    Formula: Gap_Corrected = Gap_Raw - (Slope * Age + Intercept)
    """
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    
    # Fit bias model: (Pred - True) ~ True
    gap_raw = y_pred - y_true
    reg = LinearRegression().fit(y_true, gap_raw)
    
    # Calculate expected bias for each subject based on their age
    rtm_bias = reg.predict(y_true)
    
    # Subtract bias from the predicted value
    y_pred_rtm = y_pred - rtm_bias
    return y_pred_rtm.flatten()

# ----------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------
if __name__ == "__main__":
    print("Loading harmonized data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Initialize columns for results
    df['predicted_puberty_age'] = np.nan
    df['puberty_gap_raw'] = np.nan
    df['predicted_puberty_age_rtm'] = np.nan
    df['puberty_gap_rtm'] = np.nan # <--- THIS IS YOUR KEY VARIABLE

    model_type = "combined" # We use the best model (Hormones + PDS)

    for gender in genders:
        print(f"\n{'='*40}")
        print(f"Processing Gender {gender}...")
        print(f"{'='*40}")
        
        # 1. Define Features for this gender
        features = hormones_list + pds_dict[gender]
        
        # 2. Filter Data (Drop NaNs for features/age/family just for the calculation)
        # We need complete data to run the model
        mask = (df['ab_g_stc__cohort_sex'] == int(gender)) & \
               (df[features + ['ab_g_dyn__visit_age', 'ab_g_stc__design_id__fam']].notna().all(axis=1))
        
        # Get indices to put data back later
        idx = df[mask].index 
        
        if len(idx) == 0:
            print(f"Warning: No valid data found for Gender {gender}")
            continue

        X = df.loc[idx, features].values
        y = df.loc[idx, 'ab_g_dyn__visit_age'].values
        groups = df.loc[idx, 'ab_g_stc__design_id__fam'].values

        print(f"  Sample size: {len(X)} subjects")

        # 3. Load Saved Scaler & Model Structure
        model_path = DATA_DIR / f'final_{model_type}_abcd_{gender}_model.sav'
        scaler_path = DATA_DIR / f'final_{model_type}_abcd_{gender}_scaler.pkl'
        
        print(f"  Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            base_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            saved_scaler = pickle.load(f)

        # 4. Apply Scaler (CRITICAL: Use saved scaler, do not re-fit!)
        print("  Scaling data using saved training scaler...")
        X_scaled = saved_scaler.transform(X)

        # 5. Run 10-Fold CV for Unbiased Gap Estimation
        # We re-fit the model on 9 folds and predict on the 10th.
        # This prevents "overfitting" where the model just memorizes the training data.
        kfold = GroupKFold(n_splits=10)
        y_pred_cv = np.zeros_like(y)
        
        print(f"  Running 10-Fold Cross-Validation...")
        
        for i, (train_i, test_i) in enumerate(kfold.split(X_scaled, y, groups=groups)):
            # Clone the model structure (GAM)
            fold_model = base_model 
            
            # Fit on Train
            fold_model.fit(X_scaled[train_i], y[train_i])
            
            # Predict on Test
            y_pred_cv[test_i] = fold_model.predict(X_scaled[test_i])

        # 6. Calculate Raw Gap
        gap_raw = y_pred_cv - y
        
        # 7. Calculate RTM-Corrected Predictions & Gap
        print("  Applying Regression-to-the-Mean (RTM) correction...")
        y_pred_rtm = remove_regression_to_mean_effect(y, y_pred_cv)
        
        # Recalculate gap using RTM-corrected prediction
        gap_rtm = y_pred_rtm - y

        # 8. Save back to main dataframe
        df.loc[idx, 'predicted_puberty_age'] = y_pred_cv
        df.loc[idx, 'puberty_gap_raw'] = gap_raw
        df.loc[idx, 'predicted_puberty_age_rtm'] = y_pred_rtm
        df.loc[idx, 'puberty_gap_rtm'] = gap_rtm 

        print(f"  Done. Avg Gap (Raw): {np.mean(gap_raw):.3f}, Avg Gap (RTM): {np.mean(gap_rtm):.3f}")

    # 9. Save Final Dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SUCCESS] Final RTM-corrected dataset saved to: {OUTPUT_FILE}")
