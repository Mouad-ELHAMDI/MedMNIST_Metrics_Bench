# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 18:00:05 2025

@author: Mouad
"""

import os
import json
import csv
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import openTSNE
from umap import UMAP
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------------- Utility Functions ----------------- #

def ensure_dir(directory: str) -> None:
    """Ensure that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# ----------------- Classifier with Nested CV ----------------- #

class NestedCVClassifier:
    def __init__(self, classifier_type: str = 'svm'):
        self.classifier_type = classifier_type
        self.outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        self.inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

    def _get_classifier(self):
        if self.classifier_type == 'knn':
            return KNeighborsClassifier(n_jobs=-1)
        elif self.classifier_type == 'svm':
            return SVC()
        elif self.classifier_type == 'random_forest':
            return RandomForestClassifier(n_jobs=-1)
        elif self.classifier_type == 'xgboost':
            return XGBClassifier(n_jobs=-1)
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}. Choose from: 'knn', 'svm', 'random_forest', 'xgboost'")

    def _get_param_grid(self):
        if self.classifier_type == 'knn':
            return {
                'n_neighbors': [3, 5, 7, 9, 10, 13, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto']
            }
        elif self.classifier_type == 'svm':
            return {
                'gamma': [0.5, 0.1, 0.05, 0.01, 0.001, 0.0001],
                'C': [10, 100],
                'kernel': ['rbf']
            }
        elif self.classifier_type == 'random_forest':
            return {
                'n_estimators': [100, 200],
                'max_depth': [None, 5, 10],
                'max_features': ['sqrt', 'log2'],
                'criterion': ['gini', 'entropy']
            }
        elif self.classifier_type == 'xgboost':
            return {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        else:
            return None

    def perform_nested_cv(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[Dict[str, Any]], List[float]]:
        model = self._get_classifier()
        param_grid = self._get_param_grid()

        outer_scores = []
        best_params_list = []

        for train_idx, test_idx in self.outer_cv.split(X):
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]

            grid_search = GridSearchCV(model, param_grid, cv=self.inner_cv, scoring='accuracy')
            grid_search.fit(X_train_outer, y_train_outer)

            best_params_list.append(grid_search.best_params_)
            best_model = grid_search.best_estimator_
            outer_score = best_model.score(X_test_outer, y_test_outer)
            outer_scores.append(outer_score)

        return best_params_list, outer_scores

# ----------------- Visualization ----------------- #

def plot_results(embedding: np.ndarray, y: np.ndarray, title: str, save_path: str) -> None:
    """
    Plot embedding results and save them as high-quality images.
    
    Parameters:
        embedding (np.ndarray): Transformed 2D data.
        y (np.ndarray): Labels for coloring.
        title (str): Plot title.
        save_path (str): Path to save the plot.
    """
    try:
        plt.figure(figsize=(12, 10), dpi=300)
        unique_labels = np.unique(y)
        cmap = plt.get_cmap('jet', len(unique_labels))
        norm = mcolors.Normalize(vmin=min(unique_labels), vmax=max(unique_labels))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap=cmap, alpha=0.85, s=5)
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=cmap(norm(label)), markersize=8, label=f'Class {label}')
            for label in unique_labels
        ]
        plt.legend(handles=legend_elements, title="Classes", loc='best', fontsize=10)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1, format='png')
        plt.close()
    except Exception as e:
        print(f"An error occurred while plotting {title}: {str(e)}")

def visualize_embeddings(embeddings_dict: Dict[str, Dict[str, Any]]) -> None:
    """
    Loop through the embeddings dictionary and save plots for each method and parameter value.
    
    The expected dictionary structure is:
    {
        dataset: {
            metric: {
                'tsne': { perplexity_value: embedding_array, ... },
                'umap': { n_neighbors_value: embedding_array, ... },
                'y': labels
            }
        }
    }
    """
    ensure_dir("plots")
    for dataset, metrics in embeddings_dict.items():
        for metric, data in metrics.items():
            y = data["y"]
            for perplexity, tsne_embedding in data["tsne"].items():
                title = f"{dataset} - t-SNE ({metric}) - perplexity {perplexity}"
                save_path = os.path.join("plots", f"{dataset}_tsne_{metric}_p{perplexity}.png")
                plot_results(tsne_embedding, y, title, save_path)
            for n_neighbors, umap_embedding in data["umap"].items():
                title = f"{dataset} - UMAP ({metric}) - n_neighbors {n_neighbors}"
                save_path = os.path.join("plots", f"{dataset}_umap_{metric}_n{n_neighbors}.png")
                plot_results(umap_embedding, y, title, save_path)

# ----------------- Data Loading and Dimensionality Reduction ----------------- #

def load_csv(file_name: str, directory_path: str = "MedMNIST-V2/MedMNIST", target_column: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a CSV file and return features (X) and target variable (y).
    """
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    file_path = os.path.join(directory_path, file_name)
    try:
        data = pd.read_csv(file_path)
        X = data.iloc[:, :target_column].values
        y = data.iloc[:, target_column].values
        return X, y
    except Exception as e:
        raise RuntimeError(f"Error loading file '{file_name}': {str(e)}")

def apply_tsne(X: np.ndarray, metric: str, perplexity: float = 10) -> Any:
    """
    Apply t-SNE dimensionality reduction.
    """
    try:
        tsne = openTSNE.TSNE(perplexity=perplexity, metric=metric)
        tsne_result = tsne.fit(X)
        return tsne_result
    except Exception as e:
        print(f"t-SNE failed with perplexity {perplexity} and metric {metric}: {str(e)}")
        return None

def apply_umap(X: np.ndarray, metric: str, n_neighbors: int = 15) -> Any:
    """
    Apply UMAP dimensionality reduction.
    """
    try:
        umap_inst = UMAP(n_neighbors=n_neighbors, n_components=2, metric=metric, init='pca')
        umap_result = umap_inst.fit_transform(X)
        return umap_result
    except Exception as e:
        print(f"UMAP failed with n_neighbors {n_neighbors} and metric {metric}: {str(e)}")
        return None

def generate_embeddings(datasets: List[str],
                        metrics: List[str],
                        tsne_perplexities: List[float],
                        umap_neighbors: List[int]) -> Dict[str, Dict[str, Any]]:
    """
    Generate embeddings for each dataset, metric, and parameter configuration.
    
    Returns a nested dictionary:
      {
         dataset: {
             metric: {
                 'tsne': { perplexity: embedding, ... },
                 'umap': { n_neighbors: embedding, ... },
                 'y': labels
             }
         }
      }
    """
    embeddings_dict = {}
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        embeddings_dict[dataset] = {}
        # Load dataset once per file
        X, y = load_csv(dataset)
        for metric in metrics:
            embeddings_dict[dataset][metric] = {"tsne": {}, "umap": {}, "y": y}
            # t-SNE embeddings with different perplexities
            for perplexity in tsne_perplexities:
                print(f"Running t-SNE on {dataset} with metric {metric} and perplexity {perplexity}")
                tsne_result = apply_tsne(X, metric, perplexity=perplexity)
                embeddings_dict[dataset][metric]["tsne"][perplexity] = tsne_result
            # UMAP embeddings with different n_neighbors
            for n_neighbors in umap_neighbors:
                print(f"Running UMAP on {dataset} with metric {metric} and n_neighbors {n_neighbors}")
                umap_result = apply_umap(X, metric, n_neighbors=n_neighbors)
                embeddings_dict[dataset][metric]["umap"][n_neighbors] = umap_result
    return embeddings_dict

# ----------------- Classifier Application ----------------- #

def apply_classifiers(embeddings_dict: Dict[str, Dict[str, Any]],
                      classifier_list: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Apply classifiers (using nested CV) on each embedding configuration.
    
    Returns a nested dictionary structured as:
      {
          classifier: {
              dataset: {
                  metric: {
                      method: {   # 'tsne' or 'umap'
                          parameter_value: {'best_params': ..., 'accuracy': ...},
                          ...
                      }
                  }
              }
          }
      }
    """
    results_dict = {clf: {} for clf in classifier_list}
    for dataset, metrics_data in embeddings_dict.items():
        for metric, data in metrics_data.items():
            labels = data["y"]
            for method in ['tsne', 'umap']:
                for param, embedding in data[method].items():
                    # Skip if embedding is None (failed reduction)
                    if embedding is None:
                        continue
                    embedding_np = np.array(embedding)
                    for clf in classifier_list:
                        classifier_instance = NestedCVClassifier(classifier_type=clf)
                        best_params, accuracy = classifier_instance.perform_nested_cv(embedding_np, labels)
                        if dataset not in results_dict[clf]:
                            results_dict[clf][dataset] = {}
                        if metric not in results_dict[clf][dataset]:
                            results_dict[clf][dataset][metric] = {'tsne': {}, 'umap': {}}
                        results_dict[clf][dataset][metric][method][param] = {
                            'best_params': best_params,
                            'accuracy': accuracy
                        }
    return results_dict

# ----------------- Saving Results ----------------- #

def save_embeddings_to_csv(embeddings_dict: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """
    Flatten the embeddings dictionary and save to CSV.
    """
    rows = []
    for dataset, metrics in embeddings_dict.items():
        for metric, data in metrics.items():
            y = data.get("y")
            for method in ["tsne", "umap"]:
                for param, embedding in data[method].items():
                    if embedding is not None:
                        row = {
                            "dataset": dataset,
                            "metric": metric,
                            "method": method,
                            "parameter": param,
                            "embeddings": np.array_str(np.array(embedding))
                        }
                        rows.append(row)
                    else:
                        print(f"Warning: {method} embedding for {dataset} with metric {metric} and parameter {param} is None.")
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Embeddings saved to {output_path}")

def save_results_to_csv(results_dict: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """
    Flatten classifier results and save to CSV.
    """
    flattened_results = []
    for clf, datasets in results_dict.items():
        for dataset, metrics in datasets.items():
            for metric, methods in metrics.items():
                for method in ['tsne', 'umap']:
                    for param, res in methods[method].items():
                        flattened_results.append({
                            "classifier": clf,
                            "dataset": dataset,
                            "metric": metric,
                            "method": method,
                            "parameter": param,
                            "best_params": json.dumps(res["best_params"]),
                            "accuracy": res["accuracy"]
                        })
    df = pd.DataFrame(flattened_results)
    df.to_csv(output_path, index=False)
    print(f"Classifier results saved to {output_path}")

def extract_max_accuracies(results_dict: Dict[str, Dict[str, Any]]
                           ) -> Dict[str, Dict[str, Any]]:
    """
    Extract the maximum accuracy across all datasets, metrics, and parameter settings for each classifier and method.
    """
    global_max_accuracy = {}
    for clf in results_dict:
        global_max_accuracy[clf] = {
            'tsne': {'accuracy': -np.inf, 'metric': None, 'dataset': None, 'parameter': None},
            'umap': {'accuracy': -np.inf, 'metric': None, 'dataset': None, 'parameter': None}
        }
        for dataset, metrics in results_dict[clf].items():
            for metric, methods in metrics.items():
                for method in ['tsne', 'umap']:
                    for param, res in methods[method].items():
                        acc = res['accuracy']
                        if acc > global_max_accuracy[clf][method]['accuracy']:
                            global_max_accuracy[clf][method] = {
                                'accuracy': acc,
                                'metric': metric,
                                'dataset': dataset,
                                'parameter': param
                            }
    return global_max_accuracy

def save_max_accuracies_to_csv(global_max_accuracy: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """
    Save maximum accuracies for each classifier and method to CSV.
    """
    rows = []
    for clf, methods in global_max_accuracy.items():
        for method, res in methods.items():
            rows.append({
                "classifier": clf,
                "method": method,
                "dataset": res["dataset"],
                "parameter": res["parameter"],
                "accuracy": res["accuracy"],
                "metric": res["metric"]
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Max accuracies saved to {output_path}")

# ----------------- Main Pipeline ----------------- #

def main():
    # Create directories for outputs if they don't exist.
    ensure_dir("plots")
    ensure_dir("results")

    # Define datasets, metrics, and parameter grids.
    #MedMNIST_2D = ['pneumoniamnist', 'retinamnist', 'bloodmnist', 'dermamnist', 'pathmnist', 'organsmnist', 'organcmnist', 'organamnist']
    MedMNIST_2D = ['pneumoniamnist', 'retinamnist']
    metrics = ['euclidean', 'manhattan', 'cosine', 'correlation', 'canberra']
    tsne_perplexities = [5, 15, 30]
    umap_neighbors = [5, 15, 30]
    classifier_list = ['svm', 'knn', 'random_forest', 'xgboost']

    # Generate embeddings with various parameters.
    embeddings = generate_embeddings(MedMNIST_2D, metrics, tsne_perplexities, umap_neighbors)

    # Save visualizations.
    visualize_embeddings(embeddings)

    # Apply classifiers on each embedding.
    results = apply_classifiers(embeddings, classifier_list)

    # Save results and embeddings.
    save_results_to_csv(results, os.path.join("results", "all_classifier_results.csv"))
    max_accuracies = extract_max_accuracies(results)
    save_max_accuracies_to_csv(max_accuracies, os.path.join("results", "max_accuracies.csv"))
    save_embeddings_to_csv(embeddings, os.path.join("results", "embeddings.csv"))

if __name__ == '__main__':
    main()
