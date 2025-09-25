import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

DFT_SECONDS_PER_VFE = 7200.0  

def main():
    # Load
    df = pd.read_csv('vfe_dataset_with_pure.csv')

    # Composition string for matminer
    elements = ['Mo', 'Nb', 'Ta', 'V', 'W']
    def comp_str(row):
        return ''.join(f"{el}{row[f'Comp_{el}']:.3f}" for el in elements)
    df['composition_str'] = df.apply(comp_str, axis=1)

    # matminer Magpie (composition-level)
    df['composition'] = df['composition_str'].apply(lambda x: Composition(x))
    ep = ElementProperty.from_preset('magpie')
    df = ep.featurize_dataframe(df, col_id='composition')

    # Features (DFT-free: no Pure_VFE)
    comp_cols = [c for c in df.columns if c.startswith('Comp_')]
    host_cols = [c for c in df.columns if c.startswith('Host_')]
    magpie_cols = ep.feature_labels()
    feature_cols = comp_cols + host_cols + magpie_cols

    X = df[feature_cols].values
    y = df['Alloy_VFE'].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # RF + hyperparam search
    rf_model = RandomForestRegressor(random_state=42)
    param_dist = {
        'n_estimators': [200, 400, 600, 800],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    random_search = RandomizedSearchCV(
        rf_model, param_distributions=param_dist, n_iter=30,
        scoring='neg_root_mean_squared_error', cv=5,
        verbose=2, n_jobs=-1, random_state=42
    )

    # ---- timing: tuning ----
    t0 = time.perf_counter()
    print("Starting hyperparameter tuning for Random Forest...")
    random_search.fit(X_train_scaled, y_train)
    t1 = time.perf_counter()

    print("Best parameters found:", random_search.best_params_)
    print(f"Best CV RMSE: {-random_search.best_score_:.4f} eV")

    # Train best model
    best_params = random_search.best_params_
    rf_best = RandomForestRegressor(**best_params, random_state=42)

    # ---- timing: final fit ----
    t2 = time.perf_counter()
    rf_best.fit(X_train_scaled, y_train)
    t3 = time.perf_counter()

    # Predict & evaluate
    # (warm-up to avoid measuring any one-time overhead)
    _ = rf_best.predict(X_test_scaled[:1])

    # ---- timing: inference ----
    n = X_test_scaled.shape[0]
    t4 = time.perf_counter()
    y_pred = rf_best.predict(X_test_scaled)
    t5 = time.perf_counter()
    infer_seconds = max(t5 - t4, 1e-9)
    per_comp_ms = (infer_seconds / n) * 1e3
    throughput = n / infer_seconds  # comps/s

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)

    print(f"\n=== Accuracy (RF) ===")
    print(f"Test RMSE: {rmse:.4f} eV")
    print(f"Test R²:   {r2:.4f}")
    print(f"Test MAE:  {mae:.4f} eV")

    # ---- timing summary & DFT speed-up ----
    tuning_time_s = t1 - t0
    fit_time_s    = t3 - t2
    total_train_s = tuning_time_s + fit_time_s
    dft_vs_ml_speedup = DFT_SECONDS_PER_VFE / (infer_seconds / n)  

    print("\n=== Timing (RF) ===")
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
    results_df.to_csv('vfe_rf_enhanced_results.csv', index=False)
    print("\nSaved results to 'vfe_rf_enhanced_results.csv'.")

    # Parity plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
    lo, hi = y_test.min(), y_test.max()
    plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2)
    plt.xlabel('Actual Vacancy Formation Energy (eV)')
    plt.ylabel('Predicted Vacancy Formation Energy (eV)')
    plt.title('Random Forest Tuned with Enhanced Features')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
