"""
# ----------  Created by "Mohamed Kouider Amar"    ----------%
# ----------  Email: kouideramar.mohamed@univ-medea.dz ------%
# ---- Github: https://github.com/Kouideramar/GPTree.git ----%
# -----------------------------------------------------------%

"""

import pandas as pd
import numpy as np
from GPTree import (
    distance_corr_score,
    mutual_information,
    f_statistic,
    optimize_l,
    initialize_population,
    evolve,
    fitness_function,
    print_tree,
    evaluate_final_model,
)

# --- 1. Data Loading & Preprocessing ---
# Example: Load a CSV file (replace with your actual data loading method)
# For demonstration, this assumes your data is in a CSV named 'sample_data.csv'
# and the target variable is named 'target'

# Example loading from csv, assuming the separator is a comma
file_path = 'sample_data.csv'
df = pd.read_csv(file_path)

# Example: Define features (X) and the target variable (y)
# Assuming 'target' column is the target and the other columns are features
target_variable = 'target'
X = df.drop(target_variable, axis=1).values # Features as numpy array
y = df[target_variable].values # Target variable as numpy array
feature_names = list(df.drop(target_variable, axis=1).columns) # List of feature names

l_values = np.linspace(0.01, 0.05, 10)  # Values from 0.1% - 5% and 10 equally spaced values

metrics = {
    'distance_corr': optimize_l(distance_corr_score, X, y, l_values),
    'mutual_info': optimize_l(mutual_information, X, y, l_values),
    'f_stat': optimize_l(f_statistic, X, y, l_values),
}

# Verify metrics
print("Metrics:")
for key, value in metrics.items():
    print(f"{key}: best L = {value['best_l']}, selected features size= {len(value['selected_features'])}")


# --- 2. Define Parameters ---
generations = 30
max_depth = 3
tournament_size = 7
population_size = 10

# --- 3. Run Feature Selection ---
population = initialize_population(size=population_size, metrics=metrics, max_depth=max_depth)
best_population = evolve(population, metrics, X, y, generations=generations, max_depth=max_depth, tournament_size = tournament_size)

# Evaluate Best Individual
best_individual = max(best_population, key=lambda ind: fitness_function(ind, metrics, X, y)[0])
selected_features = list(best_individual.evaluate(metrics))

selected_feature_names = [feature_names[i] for i in selected_features]
print("Selected Features:", selected_feature_names)

# Get the names of the selected features
selected_feature_names = [feature_names[i] for i in selected_features]

# Get the values of the selected features from the entire dataset (X)
selected_feature_values = X[:, selected_features]

# Create a DataFrame to store the selected feature names and their values
selected_features_df = pd.DataFrame(selected_feature_values, columns=selected_feature_names)

# --- Save Selected Features to Excel ---
# Example: Standard output file path
output_path = 'selected_features.xlsx'
selected_features_df.to_excel(output_path, index=False)
print(f"Selected features and their values saved to {output_path}")


# Print the best tree structure
print("Best Individual Tree Structure:")
print_tree(best_individual)

# --- Evaluate Final Model with 20-Fold CV ---
n_folds = 20  # Set number of folds for CV
cv_rmse_scores = evaluate_final_model(X, y, selected_features, n_folds=n_folds)

# Print the cross-validation RMSE scores
print("Cross-Validation RMSE Scores:", cv_rmse_scores)
print("Mean CV RMSE:", np.mean(cv_rmse_scores))
print("Standard Deviation of CV RMSE:", np.std(cv_rmse_scores))
