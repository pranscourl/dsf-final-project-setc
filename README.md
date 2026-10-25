# Diabetes Risk Screening Project

Course project for CSE 201: Data Science Fundamentals, Pamantasan ng Lungsod ng Pasig.

The project explores logistic regression on the 2015 Diabetes Health Indicators survey and serves the selected model through a Streamlit screening app. The data is based on the [Diabetes Health Indicators dataset on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), derived from the U.S. Behavioral Risk Factor Surveillance System (BRFSS).

## What the model estimates

The target is `Diabetes_binary`: `1` means the respondent reported a diabetes diagnosis; `0` groups respondents reported as not having diabetes, including prediabetes and diabetes only during pregnancy. The model estimates similarity to survey respondents who reported a diagnosis. It does **not** diagnose diabetes or predict who will develop it in the future.

The cleaned, balanced dataset has 70,194 rows. The fixed split used for evaluation has 56,155 real training respondents and 14,039 real test respondents. A separate 200,000-row synthetic training set is retained for comparison; it is not used as the app's default model.

The app uses **M1**, the 70k real-data logistic regression model, with 14 questions. On the project test set, M1 reached 85.01% recall and 60.21% specificity. Follow-up experiments tested a feature addition, a new 200k CART-synthetic dataset, and L2-regularized logistic regression; none improved the agreed recall-and-specificity rule, so M1 remains the app model. The balanced test set makes accuracy and precision higher than they may be in a population with lower diabetes prevalence. The notebooks also report an approximate precision adjustment at 14% prevalence; this is not a calibrated personal probability.

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

The app reports a model screening flag, not a diagnosis. Email is optional and disabled until configured. To enable it, create the config folder, copy `application/secrets.toml.example` to `application/.streamlit/secrets.toml`, fill in an SMTP app password, and keep the secrets file private. `.gitignore` excludes the expected config path and the former misplaced `application/secrets.toml` path.

For household income, users may enter either a BRFSS U.S. dollar band or an annual Philippine-peso amount. The peso amount is converted at the BSP reference rate configured in `application/risk_model.py` (currently ₱63.419 per US$1, dated 23 October 2026) and then mapped to the original BRFSS dollar band. This is a nominal currency conversion only; it does not adjust for Philippine living costs or the fact that the survey cutoffs come from 2015. Update the configured rate from the [BSP daily reference-rate table](https://www.bsp.gov.ph/statistics/external/day99_data.aspx) before future demonstrations. This conversion does not make the U.S.-trained model validated for the Philippines.

OpenAI-generated guidance is also optional. Add an API key under the `[openai]` section in the same local `secrets.toml` file and restart the app. The checkbox is off by default. If selected, the screening result and limited answer-based suggestions are sent to OpenAI to draft an encouraging note; the full answer list, name, and email are not included in that API request. API usage may incur charges separate from ChatGPT. If AI is not configured, or a request fails, the app uses its local guidance instead. AI text is general information and can be wrong; it is not a diagnosis or treatment advice.

```powershell
New-Item -ItemType Directory -Force application\.streamlit
Copy-Item application\secrets.toml.example application\.streamlit\secrets.toml
```

## Repository layout

| Path | Purpose |
|---|---|
| `application/` | Streamlit UI, model-loading and input conversion, local guidance, optional email and consent-based OpenAI personalization, and saved M1/M2 bundles. |
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
| `notebooks/10_model_experiments/` | Three follow-up experiments: training-only feature CV, a 200k CART-synthetic candidate, and regularized logistic regression, with their reports in `results/`. |
| `scripts/retrain_and_compare_models.py` | Optional command-line rebuild of the saved 70k/200k bundles and their comparison CSVs. It overwrites files in `application/models/`; do not run it unless you intend to rebuild those artifacts. |
| `requirements.txt` | Python dependencies; scikit-learn is pinned to the version used to save the model bundles. |

The `data/02_clean/diabetes_3categories_clean.csv` artifact is already recoded to the binary target; its filename records the source variant. Duplicate removal also means `diabetes_2categories5050_clean.csv` is no longer exactly balanced. The final balanced dataset used for the model split is `data/04_clean+merged_balanced(cmb)/diabetes_final_clean_merged_balanced.csv`.

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
10. The follow-up experiments are in `notebooks/10_model_experiments/`: `01_training_cv_experiments.ipynb` is training-only; `02_quality_synthetic_200k.ipynb` and `03_regularized_logreg.ipynb` make documented final comparisons on the fixed test set after training-only selection.

The notebook CSVs allow the app and notebook 10 to use the saved partitions without rebuilding the database. Notebook 10 uses the real training CSV for cross-validation. Its two later experiments score selected candidates on the shared fixed test CSV only after their training choices are set. Because that test set has now been reused for several model comparisons, treat those results as exploratory rather than as a fresh independent validation.

## Current model experiments

Notebook 10 contains three follow-up studies. The feature-addition notebook uses training-only nested CV. The CART-synthetic experiment reached 85.31% recall but reduced specificity to 59.07%, below M1's 60.21%. The regularized logistic-regression experiment selected L2 `C=0.1` from training-only CV; its final test recall was 84.998% and specificity matched M1 at 60.21%. Neither candidate met the success rule, so M1 remains the app model.

These experiments reused the same fixed test set for final comparisons. Their held-out scores should be described as exploratory comparisons on this dataset, not as independent validation. Any future model work should use training-only validation and, for a genuinely fresh final evaluation, new real respondents.

## Limitations

- Answers are self-reported and come from a 2015 U.S. survey; results may not transfer to other years or populations.
- The target is reported diagnosed diabetes, not every person who currently has undiagnosed diabetes and not future diabetes.
- The fixed test set is balanced, so its accuracy and precision do not represent the lower-prevalence population directly.
- The 14% prevalence adjustment is approximate; the app does not present it as an individual probability.
- Recall, specificity, precision and subgroup metrics describe model performance on these data. They are not a clinical validation or a substitute for a blood test or health professional.

## Tools

The repository targets Python 3.14 and pins scikit-learn to 1.9.1. `requirements.txt` includes the packages used by the app and notebooks. Use a VS Code Python/Jupyter environment for notebook work. The Streamlit app can run separately from the MariaDB-backed notebooks.
