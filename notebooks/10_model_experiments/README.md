# Notebook 10: Model experiments

This folder contains follow-up experiments that use training-only validation before final comparisons on the fixed test set. The same test set has now been used across several experiments, so these scores are exploratory comparisons rather than fresh independent validation.

## Notebooks

1. `01_training_cv_experiments.ipynb` compares a limited feature addition with the current M1 model using training-only cross-validation.
2. `02_quality_synthetic_200k.ipynb` tests a 200,000-row candidate using 100,000 real no-diabetes respondents, all eligible real diabetes respondents, and CART-generated diabetes profiles. It checks that the fixed test respondents are removed from the larger unbalanced export before training, uses the shared saved M1/M2 feature list, and performs nested CV before the final shared-test comparison.
3. `03_regularized_logreg.ipynb` compares five predeclared L2 strengths with the unregularized model. Nested CV selects candidates and thresholds using training rows only; the chosen candidate is scored once against M1 on the fixed test set.

The saved M1 and M2 bundles both use these 14 features: `HighBP`, `HighChol`, `BMI`, `Stroke`, `HeartDiseaseorAttack`, `Fruits`, `HvyAlcoholConsump`, `GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk`, `Sex`, `Age`, and `Income`.

## 200k CART experiment result

With seed 42, the final candidate reached 85.31% recall and 59.07% specificity on the fixed test set. M1 reached 85.01% recall and 60.21% specificity. CART's specificity difference from M1 was -1.14 percentage points (paired bootstrap 95% interval: -1.63 to -0.67 points), so this candidate did not meet the group's success rule. Its real-versus-synthetic discriminator ROC AUC was 0.623, indicating that generated profiles remained distinguishable from held-out real profiles.

This result shows that this synthetic-data method and recipe did not improve on M1. It does not establish that every possible synthetic-data method must fail. For this project, keep M1 as the app model unless a future training-only experiment supports a change.

Run the notebook with the repository environment and its working directory set to this folder. It reads CSV exports from `../../data/` and saved bundles from `../../application/models/`; it does not write models into the app folder.

## Final regularization experiment

With seed 42, training-only OOF selection chose `C=0.1`. The selected L2 model reached 84.998% recall and 60.21% specificity on the fixed test set, compared with M1 at 85.01% recall and 60.21% specificity. The paired 95% interval for the specificity difference was -0.057 to +0.057 percentage points. This candidate narrowly missed the recall requirement and did not improve specificity, so keep M1 as the app model.

The notebook writes nested-CV folds and summary, the full training OOF grid, the final test comparison, and paired-bootstrap differences to `results/`.
