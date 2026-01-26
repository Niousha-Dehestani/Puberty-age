"""
Script: 01_clean_data.py
Description: Integrates ABCD demographics, PDS, CBCL, and Hormone data. 
             Calculates residualized hormones (TST, DHEA) correcting for confounders.
             Stratifies participants into TD (Healthy) and ATD (Unhealthy).
             
Usage: python 01_clean_data.py --data_dir /path/to/abcd/phenotype --output_dir ./processed
Python Version: 3.10+
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

def create_id_wave_index(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to create a unified 'ID-wave' index for joining."""
    df['ID-wave'] = df['participant_id'].astype(str) + '_' + df['session_id'].astype(str)
    return df.set_index('ID-wave')

def get_medical_status(screen_path: Path) -> tuple[list, list]:
    """Identifies healthy and unhealthy participants based on medical conditions."""
    screen_df = pd.read_parquet(screen_path)
    med_cols = [c for c in screen_df.columns if 'ab_p_screen__med_' in c and c <= 'ab_p_screen__med_017']
    screen_df = screen_df.set_index('participant_id')[med_cols]
    
    has_condition = (screen_df[med_cols] == '1').any(axis=1)
    unhealthy_ids = screen_df[has_condition].index.tolist()
    healthy_ids = screen_df[~has_condition].index.tolist()
    return healthy_ids, unhealthy_ids

def load_and_merge_demographics(data_dir: Path) -> pd.DataFrame:
    """Loads and merges Demographics, PDS, and CBCL using ID-wave."""
    print("Loading Demographics, PDS, and CBCL data...")

    dyn_df = create_id_wave_index(pd.read_parquet(data_dir / 'ab_g_dyn.parquet'))
    stc_df = pd.read_parquet(data_dir / 'ab_g_stc.parquet').set_index('participant_id')

    demo_df = dyn_df[['participant_id', 'ab_g_dyn__visit_age', 'ab_g_dyn__design_site']].join(
        stc_df[['ab_g_stc__cohort_sex', 'ab_g_stc__design_id__fam', 'ab_g_stc__cohort_ethn', 'ab_g_stc__cohort_race__nih']],
        on='participant_id', how='left'
    )

    pds_cols = ['participant_id', 'session_id', 'ph_p_pds__f_mean', 'ph_p_pds__m_mean', 'ph_p_pds_age', 'ph_p_pds_001', 'ph_p_pds_002', 'ph_p_pds_003']
    pds_df = create_id_wave_index(pd.read_parquet(data_dir / 'ph_p_pds.parquet')[pds_cols]).drop(columns=['participant_id', 'session_id']) 

    cbcl_cols = [c for c in pd.read_parquet(data_dir / 'mh_p_cbcl.parquet').columns if 'mh_p_cbcl__dsm' in c] + ['participant_id', 'session_id']
    cbcl_df = create_id_wave_index(pd.read_parquet(data_dir / 'mh_p_cbcl.parquet')[cbcl_cols]).drop(columns=['participant_id', 'session_id'])

    return pd.concat([demo_df, pds_df, cbcl_df], axis=1, join='inner')

def classify_participants(df: pd.DataFrame, healthy_ids: list, unhealthy_ids: list) -> pd.DataFrame:
    """Classifies participants as TD or ATD."""
    cbcl_cols = [c for c in df.columns if 'cbcl__dsm' in c]
    df = df.dropna(subset=cbcl_cols, how='all').copy()

    df['max_cbcl_tscore'] = df[cbcl_cols].max(axis=1)
    is_med_healthy = df['participant_id'].isin(healthy_ids)
    is_med_unhealthy = df['participant_id'].isin(unhealthy_ids)
    is_cbcl_low = df['max_cbcl_tscore'] <= 60
    is_cbcl_high = df['max_cbcl_tscore'] > 60

    df['clinical_group'] = 'Unassigned'
    df.loc[is_med_healthy & is_cbcl_low, 'clinical_group'] = 'TD'
    df.loc[is_med_unhealthy | is_cbcl_high, 'clinical_group'] = 'ATD'

    # Remove incomplete ghost participants
    return df[df['clinical_group'] != 'Unassigned']

def load_and_clean_hormones(data_dir: Path) -> pd.DataFrame:
    """Loads hormone data, performs QC, and calculates regression residuals."""
    print("Loading and cleaning Hormone data (this may take a moment)...")
    hormone = pd.read_parquet(data_dir / 'ph_y_phs.parquet')
    
    # Standardize Index to match the main dataset
    hormone['ID-wave'] = hormone['participant_id'].astype(str) + '_' + hormone['session_id'].astype(str)
    hormone = hormone.set_index('ID-wave')

    # QC: Drop rows with compromised saliva notes
    qualities = ['ph_y_phs_005___1', 'ph_y_phs_005___2', 'ph_y_phs_005___3', 'ph_y_phs_005___4', 'ph_y_phs_005___5', 'ph_y_phs_005___6']
    hormone.loc[:, "hormone_notes_ss"] = hormone[qualities].astype(float).fillna(0).sum(axis=1)
    rownums = hormone["hormone_notes_ss"] > 0

    # Clean DHEA
    hormone.loc[:, "filtered_dhea_rep1"] = hormone["ph_y_phs__dhea__r01_qnt"]
    hormone.loc[:, "filtered_dhea_rep2"] = hormone["ph_y_phs__dhea__r02_qnt"]
    hormone.loc[hormone["ph_y_phs__dhea__r01_null"] == 1, "filtered_dhea_rep1"] = 0
    hormone.loc[hormone["ph_y_phs__dhea__r02_null"] == 1, "filtered_dhea_rep2"] = 0

    rownums_rep1 = (hormone["ph_y_phs__dhea__r01_qnt"] < 5) | (hormone["ph_y_phs__dhea__r01_qnt"] > 1000)
    rownums_rep2 = (hormone["ph_y_phs__dhea__r02_qnt"] < 5) | (hormone["ph_y_phs__dhea__r02_qnt"] > 1000)
    hormone.loc[(rownums & rownums_rep1), "filtered_dhea_rep1"] = np.nan
    hormone.loc[(rownums & rownums_rep2), "filtered_dhea_rep2"] = np.nan
    hormone["dhea_filter"] = hormone[["filtered_dhea_rep1", "filtered_dhea_rep2"]].mean(axis=1)

    # Clean TST (ERT)
    hormone.loc[:, "filtered_tst_rep1"] = hormone["ph_y_phs__ert__r01_qnt"]
    hormone.loc[:, "filtered_tst_rep2"] = hormone["ph_y_phs__ert__r02_qnt"]
    hormone.loc[hormone["ph_y_phs__ert__r01_null"] == 1, "filtered_tst_rep1"] = 0
    hormone.loc[hormone["ph_y_phs__ert__r02_null"] == 1, "filtered_tst_rep2"] = 0

    rownums_rep1 = (hormone["ph_y_phs__ert__r01_qnt"] < 5) | (hormone["ph_y_phs__ert__r01_qnt"] > 1000)
    rownums_rep2 = (hormone["ph_y_phs__ert__r02_qnt"] < 5) | (hormone["ph_y_phs__ert__r02_qnt"] > 1000)
    hormone.loc[(rownums & rownums_rep1), "filtered_tst_rep1"] = np.nan
    hormone.loc[(rownums & rownums_rep2), "filtered_tst_rep2"] = np.nan
    hormone["tst_filter"] = hormone[["filtered_tst_rep1", "filtered_tst_rep2"]].mean(axis=1)

    # Clean Times
    times = ['ph_y_phs__coll__end_t', "ph_y_phs_001", "ph_y_phs__coll_t", "ph_y_phs__freeze_t"]
    hormone = hormone.dropna(subset=times).copy()

    for time in times:
        hormone[time] = pd.to_datetime(hormone[time], format="mixed", errors="coerce")
        hormone[f"{time}_saliva_filter"] = hormone[time].dt.hour * 60 + hormone[time].dt.minute

    # Regress out Confounders
    confounders = ['ph_y_phs_001_saliva_filter', 'ph_y_phs__freeze_t_saliva_filter', 'ph_y_phs_004', 'ph_y_phs_003']
    hormone = hormone.dropna(subset=confounders + ['tst_filter', 'dhea_filter']).copy()

    X = hormone[confounders].astype(float) 

    for target in ['tst_filter', 'dhea_filter']:
        y = hormone[target].astype(float)
        reg = LinearRegression().fit(X, y)
        hormone[f"{target}_rsi_hormone"] = y - reg.predict(X)

    return hormone[['tst_filter_rsi_hormone', 'dhea_filter_rsi_hormone']]


if __name__ == "__main__":
    # Set up command-line arguments for GitHub users
    parser = argparse.ArgumentParser(description="Clean ABCD 6.0 data for Puberty Age Gap model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw ABCD 6.0 phenotype folder")
    parser.add_argument("--output_dir", type=str, default="./processed", help="Path to save clean data")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Get Base Data
    healthy_list, unhealthy_list = get_medical_status(DATA_DIR / 'ab_p_screen.parquet')
    merged_data = load_and_merge_demographics(DATA_DIR)
    
    # 2. Classify (TD/ATD)
    classified_data = classify_participants(merged_data, healthy_list, unhealthy_list)

    # 3. Get Residualized Hormones
    hormone_data = load_and_clean_hormones(DATA_DIR)

    # 4. FINAL MERGE (Inner join: keeps only participants with BOTH behavior and valid hormone data)
    print("\nMerging final data with hormones...")
    final_combined = pd.concat([classified_data, hormone_data], axis=1, join='inner')

    # Print Final Sample Size
    print(f"\nFINAL SAMPLE SIZE (with Hormones): {len(final_combined)} participants")
    print(final_combined['clinical_group'].value_counts())

    # 5. Save
    output_path = OUTPUT_DIR / '01_final_cleaned_data_with_hormones.csv'
    final_combined.to_csv(output_path)
    print(f"\nSuccess! Final clean data saved to {output_path}")
