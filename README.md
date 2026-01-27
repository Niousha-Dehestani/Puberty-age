# Puberty Age Gap (PAG): A Novel Metric of Pubertal Timing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository hosts the scripts and code used for our manuscript, **"“Puberty age gap”: new method of assessing pubertal timing and its association with mental health problems."** The study focuses on deriving a novel metric of pubertal timing, the Puberty Age Gap (PAG), with applications to developmental neuroimaging and adolescent mental health.

## 📂 Repository Structure

The repository is organized into three main analysis steps, plus a data folder for templates.

```text
├── data/
│   ├── template_data_female.csv      # Empty template with required columns for females
│   └── template_data_male.csv        # Empty template with required columns for males
│
├── scripts/
│   ├── 01_clean_data/
│   │   ├── 01_clean_data.py          # Initial cleaning of PDS and hormone data
│   │   └── 01b_harmonize_data.py     # Merges and harmonizes data for analysis
│   │
│   ├── 02_model_development/
│   │   ├── 02_run_models.py          # Trains the Generalized Additive Models (GAMs)
│   │   └── prediction.py             # Helper functions for model prediction
│   │
│   └── 03_downstream_tasks/
│       └── 03a_calculate_puberty_gap.py  # Calculates final PAG and applies RTM correction


To use these scripts with your own data, your input files must match the structure required by the pipeline.

We have provided empty template files in the `data/` folder (`template_data_male.csv` and `template_data_female.csv`) containing the specific column headers required.

**Required Columns:**

* `src_subject_id`: Participant ID
* `ab_g_dyn__visit_age`: Age in months
* `ab_g_stc__design_id__fam`: Family or Site ID (used for cross-validation grouping)
* `tst_filter_rsi_hormone`: Testosterone levels
* `dhea_filter_rsi_hormone`: DHEA levels
* `ph_p_pds_*`: Specific Pubertal Development Scale items (refer to the templates for sex-specific items)

## Analysis Pipeline

The code is organized into distinct analytical steps:

### 1. Data Cleaning and Preprocessing (`scripts/01_clean_data/`)
* **`01_clean_data.py`**: Processes and cleans raw Pubertal Development Scale (PDS) questionnaire data and biospecimen hormone data. Handles missing values and initial quality control.
* **`01b_harmonize_data.py`**: Harmonizes the cleaned datasets into a unified format ready for modeling.

### 2. Puberty Age Model Development (`scripts/02_model_development/`)
* **`02_run_models.py`**:
    * **Algorithm:** Implements Generalized Additive Models (GAMs) to calculate biological "Puberty Age".
    * **Stratification:** Models are fit separately for each sex (Males vs. Females) to account for distinct developmental trajectories.
    * **Data Splitting:** Randomly splits the cleaned data into training and test sets to ensure model validation and prevent overfitting.
* **`prediction.py`**: Contains helper functions used for generating model predictions.

### 3. Metric Derivation (`scripts/03_downstream_tasks/`)
* **`03a_calculate_puberty_gap.py`**:
    * Generates unbiased predictions using 10-Fold Cross-Validation.
    * Calculates the **Puberty Age Gap (PAG)** by subtracting chronological age from the model-predicted Puberty Age.
    * **RTM Correction:** Applies a Regression-to-the-Mean (RTM) correction to remove age-related bias from the final metric.

## Citation

If you use this repository, metric, or code in your research, please cite our manuscript:

**Dehestani, N.**, Vijayakumar, N., Ball, G., Mansour L, S., Whittle, S., & Silk, T. J. (2024). “Puberty age gap”: new method of assessing pubertal timing and its association with mental health problems. *Molecular Psychiatry, 29*(2), 221-228.

## Contact

For questions regarding the methodology or code, please contact:
**Niousha Dehestani** – [newsha.dehestani.95@gmail.com](mailto:newsha.dehestani.95@gmail.com)
