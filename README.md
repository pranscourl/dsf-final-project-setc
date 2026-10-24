# Diabetes Risk Screening Project

Course project for CSE 201: Data Science Fundamentals, Pamantasan ng Lungsod ng Pasig.

The project explores logistic regression on the 2015 Diabetes Health Indicators survey and serves the selected model through a Streamlit screening app. The data is based on the [Diabetes Health Indicators dataset on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), derived from the U.S. Behavioral Risk Factor Surveillance System (BRFSS).

## What the model estimates

The target is `Diabetes_binary`: `1` means the respondent reported a diabetes diagnosis; `0` groups respondents reported as not having diabetes, including prediabetes and diabetes only during pregnancy. The model estimates similarity to survey respondents who reported a diagnosis. It does **not** diagnose diabetes or predict who will develop it in the future.

The cleaned, balanced dataset has 70,194 rows. The fixed split used for evaluation has 56,155 real training respondents and 14,039 real test respondents. A separate 200,000-row synthetic training set is retained for comparison; it is not used as the app's default model.

The current app uses **M1**, the 70k real-data logistic regression model, with 14 questions. In the latest shared-test comparison, M1 reached 85.01% recall and 60.21% specificity. A later training-only CV run did not find a reason to replace M1 with the tested feature addition, so M1 remains the app model. The balanced test set makes accuracy and precision higher than they may be in a population with lower diabetes prevalence. The notebooks also report an approximate precision adjustment at 14% prevalence; this is not a calibrated personal probability.

The current working success rule is:

1. Require recall of at least 85%.
2. Compare eligible models primarily by specificity.
3. Report precision on the balanced test set and the approximate 14%-prevalence adjustment.

## Run the app

The app loads its saved model from `application/models/` and does not need MariaDB to start.

From the repository root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Set-Location application
streamlit run app.py
```

Open the local address Streamlit prints, usually <http://localhost:8501>. The app selects `logreg_70k.joblib` through `DEFAULT_MODEL` in `application/risk_model.py`.

### Manually exercise the app

1. Complete all 14 questions with a fictional or sample profile. Use a plausible height and weight so the computed BMI is within the supported range.
2. Confirm the progress indicator changes as answers are completed.
3. Submit and confirm the result and next-step guidance appear.
4. Change an answer and confirm the previous result clears; submit again to get the updated result.
5. Try both metric and imperial height/weight entry. Leave the optional email blank unless SMTP is deliberately configured.

The app reports a model screening flag, not a diagnosis. Email is optional and disabled until configured. To enable it, create the config folder, copy `application/secrets.toml.example` to `application/.streamlit/secrets.toml`, fill in an SMTP app password, and keep the secrets file private. `.gitignore` excludes that file.

```powershell
New-Item -ItemType Directory -Force application\.streamlit
Copy-Item application\secrets.toml.example application\.streamlit\secrets.toml
```

## Repository layout

| Path | Purpose |
|---|---|
| `application/` | Streamlit UI, model-loading and input conversion, guidance, optional email, and saved M1/M2 bundles. |
| `data/01_original/` | Three original Kaggle CSV variants: three-category, binary balanced, and binary unbalanced. |
| `data/02_clean/` | Cleaned versions created by the transformation notebooks. |
| `data/03_clean+merged_unbalanced(cmu)/` | Deduplicated, merged unbalanced dataset. |
| `data/04_clean+merged_balanced(cmb)/` | Deduplicated, balanced dataset used to create the 70k training and test split. |
| `data/05_cmb_train_test_for_models/` | Fixed real training and held-out test CSVs shared by the models. |
| `data/06_cmb_logreg_70kdataset/` | M1 evaluation results. |
| `data/07_cmu+synthetic/` | Synthetic-augmented training CSV and the real-only control comparison. The training CSV labels generated rows with `is_synthetic`; it contains no test rows. |
| `data/08_model_comparison/` | Results from notebook 08's model comparison. |
| `notebooks/01`–`08` | Extraction, transformation, loading, EDA, initial modeling, regression coursework, synthetic modeling, and the original model comparison. |
| `notebooks/09_utils/synth_utils.py` | Shared survey-aware synthetic-row generator used by the synthetic-model workflow. |
| `notebooks/10_model_experiments/` | Nested training-only CV comparison of M1 with a limited feature addition; it keeps the test set closed when CV does not support a change. Its CV reports are in `results/`. |
| `scripts/retrain_and_compare_models.py` | Optional command-line rebuild of the saved 70k/200k bundles and their comparison CSVs. It overwrites files in `application/models/`; do not run it unless you intend to rebuild those artifacts. |
| `requirements.txt` | Python dependencies; scikit-learn is pinned to the version used to save the model bundles. |

The model file in `notebooks/06_multilinearreg/linprob_70k.joblib` is a coursework reference artifact for notebook 08; the app does not load it.

## Reproduce the database-backed notebooks

The extraction, transformation, EDA and original modeling notebooks use a local MariaDB database named `diabetes_final_db`. They connect with `mysql+pymysql://root:@localhost:3306/diabetes_final_db` by default (user `root`, no password). If your local MariaDB credentials differ, update the connection URL in the relevant notebooks before running them.

1. Start MariaDB and create the database if it does not exist:

   ```sql
   CREATE DATABASE diabetes_final_db;
   ```

2. Run the three notebooks in `notebooks/01_extracted_data/` to load the source CSVs as `diabetes_3categories`, `diabetes_2categories5050`, and `diabetes_2categories_unbalanced`.
3. Run the three transformation notebooks in `notebooks/02_transform_data/` to create the corresponding `_clean` tables.
4. Run `notebooks/03_load_data/01_load_merged_datasets.ipynb` to create the merged unbalanced and balanced tables.
5. Run notebooks in `notebooks/04_eda&stats/` for data checks, the feature dictionary, EDA and statistical analysis.
6. Run the notebooks in `notebooks/05_modeling_cmb(original70K)/` in order: prepare/split, train, evaluate. This creates the shared real training/test tables and CSV files.
7. Run `notebooks/06_multilinearreg/06_multilinearreg.ipynb` for the course-required linear-probability and BMI regressions (optional for launching the app).
8. Run the notebooks in `notebooks/07_modeling_cmu(+synthetic200k)/` in order to prepare synthetic training rows, train M2 and evaluate it on the same real test set.
9. Run `notebooks/08_model_comparison/08_model_comparison.ipynb` for the original six-model comparison. Its decision rule records an earlier group proposal; the current agreed metric and follow-up experiments are in notebook 10.
10. Run `notebooks/10_model_experiments/01_training_cv_experiments.ipynb` for the current, limited candidate study. Only open the held-out test for a final comparison if training-only CV supports a new candidate.

The notebook CSVs allow the app and notebook 10 to use the saved partitions without rebuilding the database. Notebook 10 uses the real training CSV for cross-validation, then reads the held-out test CSV only in its final-comparison notebook after model and threshold choices are fixed.

## Current model experiment

Notebook 10 compares current M1 with logistic regression adding Education and Physical Activity. It uses nested, stratified cross-validation on the real training CSV; scaling and threshold selection happen within training folds. The held-out test CSV is not read because the feature candidate did not improve on M1 in this run.

In the latest CV run, M1 ranked ahead of the added-feature candidate on mean specificity, while their mean outer-fold recall estimates were both close to (but slightly below) 85%. No candidate was supported as an improvement, so the notebook does not open the held-out test and the app continues to use M1. The earlier shared-test comparison of M1 and M2 is documented in notebook 08.

## Limitations

- Answers are self-reported and come from a 2015 U.S. survey; results may not transfer to other years or populations.
- The target is reported diagnosed diabetes, not every person who currently has undiagnosed diabetes and not future diabetes.
- The fixed test set is balanced, so its accuracy and precision do not represent the lower-prevalence population directly.
- The 14% prevalence adjustment is approximate; the app does not present it as an individual probability.
- Recall, specificity, precision and subgroup metrics describe model performance on these data. They are not a clinical validation or a substitute for a blood test or health professional.

## Tools

The repository targets Python 3.14 and pins scikit-learn to 1.9.1. `requirements.txt` includes the packages used by the app and notebooks. Use a VS Code Python/Jupyter environment for notebook work. The Streamlit app can run separately from the MariaDB-backed notebooks.
