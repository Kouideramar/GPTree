"""
# ----------  Created by "Mohamed Kouider Amar"    ----------%
# ----------  Email: kouideramar.mohamed@univ-medea.dz ------%
# ---- Github: https://github.com/Kouideramar/GPTree.git ----%                               --------%
# -----------------------------------------------------------%

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression, f_regression
import dcor
import random
from sklearn.model_selection import cross_val_score

# --- Define Feature Selection Metrics ---
def distance_corr_score(X, y):
    return np.array([dcor.distance_correlation(X[:, i], y) for i in range(X.shape[1])])

def mutual_information(X, y):
    return mutual_info_regression(X, y, random_state=1)

def f_statistic(X, y):
    scores, _ = f_regression(X, y)
    return scores

# --- Genetic Programming Framework ---
class GPTree:
    def __init__(self, terminal=None, left=None, right=None, operator=None):
        self.terminal = terminal  # Feature selection metric
        self.left = left
        self.right = right
        self.operator = operator  # ∪, ∩, \

    def evaluate(self, metrics):
        if self.terminal:
            return set(metrics[self.terminal]['selected_features'])
        
        if not self.left or not self.right or not self.operator:
            return set()

        left_result = self.left.evaluate(metrics)
        right_result = self.right.evaluate(metrics)

        if self.operator == '∪':
            return left_result | right_result
        elif self.operator == '∩':
            return left_result & right_result
        elif self.operator == '\\':
            return left_result - right_result
        
        return set()

# --- Fitness Function ---
def fitness_function(tree, metrics, X, y, min_features=1, penalty_weight=10.0):
    selected_features = tree.evaluate(metrics)
    num_selected = len(selected_features)
    
    # Initialize rmse to a default value
    rmse = float('inf')  # Default value when no features are selected
    
    # Train the model and calculate the base fitness score
    if num_selected > 0:
        X_selected = X[:, list(selected_features)]
        model = RandomForestRegressor(random_state=1)
        model.fit(X_selected, y)
        y_pred = model.predict(X_selected)
        rmse = np.sqrt(mean_squared_error(y, y_pred))  # Use RMSE
        base_fitness = -rmse  # Maximize negative RMSE
    else:
        base_fitness = 0  # No features selected, assign a neutral base fitness
    
    # Calculate penalty for not meeting the minimum feature requirement
    if num_selected < min_features:
        penalty = (min_features - num_selected) * penalty_weight
    else:
        penalty = 0  # No penalty if the minimum requirement is met
    
    # Adjust the fitness score by subtracting the penalty
    adjusted_fitness = base_fitness - penalty
    
    return adjusted_fitness, rmse  # Return adjusted fitness and RMSE


# --- Initialize GP Population ---
def create_random_tree(metrics, max_depth, current_depth=0):
    if current_depth >= max_depth or random.random() < 0.1:  # Chance to stop growing
        terminal = random.choice(list(metrics.keys()))
        return GPTree(terminal=terminal)
    else:
        operator = random.choice(['∪', '∩', '\\'])
        left = create_random_tree(metrics, max_depth, current_depth + 1)
        right = create_random_tree(metrics, max_depth, current_depth + 1)
        return GPTree(operator=operator, left=left, right=right)

def initialize_population(size, metrics, max_depth=12):
    population = []
    for _ in range(size):
        population.append(create_random_tree(metrics, max_depth))
    return population

# --- Genetic Operations ---
def mutate(tree, metrics, max_depth):
    if random.random() < 0.3 and tree.operator:
        tree.operator = random.choice(['∪', '∩', '\\'])
        return tree
    elif random.random() < 0.6:  # Mutate operator
        if tree.left and tree.right:
            if random.random() < 0.5:
                tree.left = create_random_tree(metrics, max_depth, 0)
            else:
                tree.right = create_random_tree(metrics, max_depth, 0)
        return tree
    else:  # Mutate terminal
        tree.terminal = random.choice(list(metrics.keys()))
    return tree

def crossover(tree1, tree2, metrics, max_depth, current_depth=0):
    if current_depth >= max_depth or not tree1 or not tree2:
        return create_random_tree(metrics, max_depth)

    if random.random() < 0.5:  # Crossover subtrees
        new_left = crossover(tree1.left, tree2.left, metrics, max_depth, current_depth + 1)
        new_right = crossover(tree1.right, tree2.right, metrics, max_depth, current_depth + 1)
        operator = random.choice(['∪', '∩', '\\'])
        return GPTree(left=new_left, right=new_right, operator=operator)
    else:  # Crossover operators (less frequent)
        return create_random_tree(metrics, max_depth)

# --- Optimize L for each metric ---
def optimize_l(metric_func, X_train, y_train, l_values, random_state=1):
    best_l = None
    best_score = float('-inf')
    for l in l_values:
        if metric_func == distance_corr_score:
            scores = np.argsort(distance_corr_score(X_train, y_train))
        elif metric_func == mutual_information:
            scores = np.argsort(mutual_information(X_train, y_train))
        elif metric_func == f_statistic:
            scores = np.argsort(f_statistic(X_train, y_train))
        else:
            raise ValueError("Invalid Metric")

        selected_features = scores[-int(l * X_train.shape[1]):].tolist()

        X_selected = X_train[:, selected_features]
        model = RandomForestRegressor(random_state=random_state)
        model.fit(X_selected, y_train)
        y_pred = model.predict(X_selected)
        score = -np.sqrt(mean_squared_error(y_train, y_pred))  # Use RMSE for optimization

        if score > best_score:
            best_score = score
            best_l = l
            best_features = selected_features

    return {'best_l': best_l, 'selected_features': best_features}

# --- Evolve Population with Tournament Selection and Crowding Distance ---
def crowding_distance(population, metrics, X, y):
    if not population:
        return [0]  # Handle empty population case
    
    distances = [0] * len(population)
    fitness_scores = [fitness_function(ind, metrics, X, y)[0] for ind in population]  # Use adjusted fitness
    
    if len(population) <= 2:
        return [float('inf')] * len(population)

    # Sort population by fitness
    sorted_population_indices = sorted(range(len(population)), key=lambda k: fitness_scores[k], reverse=True)

    distances[sorted_population_indices[0]] = float('inf')
    distances[sorted_population_indices[-1]] = float('inf')

    for i in range(1, len(population) - 1):
        distances[sorted_population_indices[i]] = (fitness_scores[sorted_population_indices[i - 1]] - fitness_scores[sorted_population_indices[i + 1]])

    return distances

def tournament_selection(population, metrics, X, y, tournament_size):
    if not population:
        return []  # Handle empty population case
    
    selected = []
    distances = crowding_distance(population, metrics, X, y)

    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_individuals = [population[i] for i in tournament_indices]
        tournament_fitnesses = [fitness_function(ind, metrics, X, y)[0] for ind in tournament_individuals]  # Use adjusted fitness
        tournament_distances = [distances[i] for i in tournament_indices]

        # Combine fitness and crowding distance
        combined_scores = [(tournament_fitnesses[i], tournament_distances[i]) for i in range(tournament_size)]

        # Choose the best based on fitness and crowding distance (first sort by fitness then by crowding distance)
        winner_index = sorted(range(tournament_size), key=lambda k: combined_scores[k], reverse=True)[0]
        selected.append(tournament_individuals[winner_index])

    return selected

def evolve(population, metrics, X, y, generations=30, max_depth=12, tournament_size=7):
    for gen in range(generations):
        print(f"Generation {gen + 1}")
        
        # Selection with crowding distance
        selected_population = tournament_selection(population, metrics, X, y, tournament_size)

        new_population = []
        while len(new_population) < len(population):
            parent1, parent2 = random.sample(selected_population, 2)
            child = crossover(parent1, parent2, metrics, max_depth)
            if random.random() < 0.3:  # Apply mutation
                child = mutate(child, metrics, max_depth)
            new_population.append(child)
        
        population = new_population

    return population

def print_tree(tree, depth=0):
    indent = "  " * depth
    if tree.terminal:
        print(f"{indent}Terminal: {tree.terminal}")
    else:
        print(f"{indent}Operator: {tree.operator}")
        print_tree(tree.left, depth + 1)
        print_tree(tree.right, depth + 1)


# --- Evaluate Final Model with 5-Fold CV ---
def evaluate_final_model(X, y, selected_features, n_folds=5):
    X_selected = X[:, selected_features]
    model = RandomForestRegressor(random_state=1)
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X_selected, y, cv=n_folds, scoring='neg_root_mean_squared_error')
    
    # Convert negative RMSE to positive RMSE
    cv_rmse_scores = -cv_scores
    
    return cv_rmse_scores