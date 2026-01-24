# Puberty Age Gap (PAG): A Novel Metric of Pubertal Timing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository hosts the scripts and code used for our manuscript, *"“Puberty age gap”: new method of assessing pubertal timing and its association with mental health problems."* The study focuses on deriving a novel metric of pubertal timing, the **Puberty Age Gap (PAG)**, with applications to developmental neuroimaging and adolescent mental health.

## Overview

The repository is organized into distinct analytical steps, each corresponding to a specific phase in the computational pipeline. Below is a summary of what each step accomplishes:

### 1. Data Cleaning and Preprocessing (`code/01_data_cleaning`)
* Process and clean both Pubertal Development Scale (PDS) questionnaire data and biospecimen hormone data from participants in the ABCD Study dataset.
* Handle missing values and prepare the dataset for downstream analysis.

### 2. Puberty Age Model Development (`code/02_model_development`)
* **Data Splitting:** Randomly split the cleaned data into training and test sets to ensure model validation and prevent overfitting.
* **Algorithm:** Implement Generalized Additive Models (GAMs) to calculate biological "Puberty Age".
* **Stratification:** Models are fit separately for each sex to account for distinct male and female developmental trajectories.
* **Metric Derivation:** Calculate the Puberty-Age-Gap (PAG) by subtracting chronological age from the model-predicted Puberty Age.
