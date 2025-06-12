import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty


def main():
    
    df = pd.read_csv('vfe_dataset_with_pure.csv')

    
    elements = ['Mo', 'Nb', 'Ta', 'V', 'W']

    def comp_str(row):
        return ''.join(f"{el}{row[f'Comp_{el}']:.3f}" for el in elements)

    df['composition_str'] = df.apply(comp_str, axis=1)

    
    df['composition'] = df['composition_str'].apply(lambda x: Composition(x))


    ep = ElementProperty.from_preset('magpie')
    df = ep.featurize_dataframe(df, col_id='composition')

    
    comp_cols = [c for c in df.columns if c.startswith('Comp_')]
    host_cols = [c for c in df.columns if c.startswith('Host_')]
    magpie_cols = ep.feature_labels()  

    feature_cols = comp_cols + ['Pure_VFE'] + host_cols + magpie_cols

    X = df[feature_cols].values
    y = df['Alloy_VFE'].values

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
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

    print("Starting hyperparameter tuning for Random Forest with enhanced features...")
    random_search.fit(X_train_scaled, y_train)

    print("Best parameters found:", random_search.best_params_)
    print(f"Best CV RMSE: {-random_search.best_score_:.4f} eV")

    best_params = random_search.best_params_
    rf_best = RandomForestRegressor(**best_params, random_state=42)
    rf_best.fit(X_train_scaled, y_train)

    
    y_pred = rf_best.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.4f} eV")
    print(f"Test R²: {r2:.4f}")

    
    results_df = pd.DataFrame({
        'Actual_VFE': y_test,
        'Predicted_VFE': y_pred,
        'Percent_Diff': 100 * np.abs((y_test - y_pred) / y_test)
    })
    results_df.to_csv('vfe_rf_enhanced_results.csv', index=False)
    print("Saved results to 'vfe_rf_enhanced_results.csv'.")

    
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Vacancy Formation Energy (eV)')
    plt.ylabel('Predicted Vacancy Formation Energy (eV)')
    plt.title('Random Forest Tuned with Enhanced Features')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
