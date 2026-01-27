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


