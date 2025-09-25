import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt

from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

DFT_SECONDS_PER_VFE = 7200.0  # 2 hours per DFT calculation (baseline)

def main():
    df = pd.read_csv('vfe_dataset_with_pure.csv')

    elements = ['Mo', 'Nb', 'Ta', 'V', 'W']
    def comp_str(row):
        return ''.join(f"{el}{row[f'Comp_{el}']:.3f}" for el in elements)
    df['composition_str'] = df.apply(comp_str, axis=1)

    df['composition'] = df['composition_str'].apply(lambda x: Composition(x))

    ep = ElementProperty.from_preset('magpie')
    df = ep.featurize_dataframe(df, col_id='composition')

    comp_cols  = [c for c in df.columns if c.startswith('Comp_')]
    host_cols  = [c for c in df.columns if c.startswith('Host_')]
    mag_cols   = ep.feature_labels()

    # DFT-free inputs (no Pure_VFE)
    feature_cols = comp_cols + host_cols + mag_cols

    X = df[feature_cols].values
    y = df['Alloy_VFE'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, random_state=42)

    param_dist = {
        'max_depth': [4, 6, 8, 10, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 1, 3, 5]
    }

    random_search = RandomizedSearchCV(
        xgb_model, param_distributions=param_dist, n_iter=50,
        scoring='neg_root_mean_squared_error', cv=5,
        verbose=2, n_jobs=-1, random_state=42
    )

    # ---- timing: tuning ----
    t0 = time.perf_counter()
    print("Starting hyperparameter tuning (XGB)...")
    random_search.fit(X_train_scaled, y_train)
    t1 = time.perf_counter()

    print("Best parameters found:", random_search.best_params_)
    print(f"Best CV RMSE: {-random_search.best_score_:.4f} eV")

    best_params = random_search.best_params_
    best_params['objective'] = 'reg:squarederror'
    best_params['seed'] = 42
    best_params['n_estimators'] = 1000

    # xgb.train uses DMatrix; measure final training time separately
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dvalid = xgb.DMatrix(X_test_scaled,  label=y_test)

    # ---- timing: final fit ----
    t2 = time.perf_counter()
    bst = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dvalid, 'validation')],
        early_stopping_rounds=30,
        verbose_eval=20
    )
    t3 = time.perf_counter()

    # Warm-up
    _ = bst.predict(xgb.DMatrix(X_test_scaled[:1]))

    # ---- timing: inference ----
    n = X_test_scaled.shape[0]
    t4 = time.perf_counter()
    y_pred = bst.predict(xgb.DMatrix(X_test_scaled), iteration_range=(0, bst.best_iteration + 1))
    t5 = time.perf_counter()
    infer_seconds = max(t5 - t4, 1e-9)
    per_comp_ms = (infer_seconds / n) * 1e3
    throughput = n / infer_seconds

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)

    print(f"\n=== Accuracy (XGB) ===")
    print(f"Test RMSE: {rmse:.4f} eV")
    print(f"Test R²:   {r2:.4f}")
    print(f"Test MAE:  {mae:.4f} eV")

    # ---- timing summary & DFT speed-up ----
    tuning_time_s = t1 - t0
    fit_time_s    = t3 - t2
    total_train_s = tuning_time_s + fit_time_s
    dft_vs_ml_speedup = DFT_SECONDS_PER_VFE / (infer_seconds / n)

    print("\n=== Timing (XGB) ===")
    print(f"Tuning time:         {tuning_time_s:.3f} s")
    print(f"Final fit time:      {fit_time_s:.3f} s")
    print(f"Total train time:    {total_train_s:.3f} s")
    print(f"Inference:           {infer_seconds:.6f} s over {n} comps")
    print(f"Throughput:          {throughput:,.1f} comps/s")
    print(f"Median per-comp:     {per_comp_ms:.3f} ms (approx)")
    print(f"Estimated DFT→ML speed-up (per VFE, 2 h baseline): ~{dft_vs_ml_speedup:,.0f}×")

    # Save results
    results_df = pd.DataFrame({
        'Actual_VFE': y_test,
        'Predicted_VFE': y_pred,
        'Percent_Diff': 100 * np.abs((y_test - y_pred) / y_test)
    })
    results_df.to_csv('vfe_xgb_tuned_results.csv', index=False)
    print("\nSaved results to 'vfe_xgb_tuned_results.csv'.")

    # Parity plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
    lo, hi = y_test.min(), y_test.max()
    plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2)
    plt.xlabel('Actual Vacancy Formation Energy (eV)')
    plt.ylabel('Predicted Vacancy Formation Energy (eV)')
    plt.title('XGBoost Tuned: Actual vs Predicted VFE')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
