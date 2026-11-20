# Regression and Residual Diagnostics

Linear regression exercises focused on choosing the right transformation and checking model assumptions through residuals.

| Script | Data | What it does |
|---|---|---|
| `sulfate_loglog_regression.py` | Sulfate concentration in blood over time (21 measurements) | Fits a log-log regression, back-transforms predictions, and compares residuals in log and original coordinates |
| `body_mass_regression.py` | [Body measurements](http://www.statsci.org/data/oz/physical.html) (mass vs. 9 girth/height measurements) | Regresses mass and the cube root of mass on all measurements; compares RMSE, residual plots, residual distributions, Q-Q plots and heteroscedasticity |
| `abalone_age_regression.py` | [Abalone](https://archive.ics.uci.edu/dataset/1/abalone), UCI | Predicts age from physical measurements (ignoring sex) and inspects residuals |
| `abalone_log_age_elasticnet.py` | Abalone, UCI | Predicts log age with and without sex as a feature, then fits an ElasticNet model tuned by 5-fold grid-search cross-validation and compares all models on R² and RMSE |

## Run
Download `abalone.data` from UCI into this folder for the abalone scripts; the other two scripts load their data directly.
```bash
python sulfate_loglog_regression.py
```
