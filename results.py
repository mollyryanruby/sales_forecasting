"""
This script creates a csv of the model scores and outputs a plot to visualize
the comparison.

Models include Linear Regression, Random Forest, XGBoost, LSTM,and ARIMA.
"""

import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def create_results_df():
    """Returns a pandas dataframe with the root mean squared error, mean
    absolute error, and R2 score for each model.
    """
    # Load pickled scores for each model
    results_dict = pickle.load(open("model_scores.p", "rb"))

    # Create pandas df and save as csv
    results_df = pd.DataFrame.from_dict(results_dict, orient='index',
                                        columns=['RMSE', 'MAE', 'R2'])

    results_df = results_df.sort_values(by='RMSE',
                                        ascending=False).reset_index()

    results_df.to_csv('../data/results.csv')

    return results_df

def plot_results(results_df):
    """Generates and saves and lineplot with one line indicating RMSE scores
    for each model and one line indicating MAE scores for each model.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(np.arange(len(results_df)), 'RMSE', data=results_df, ax=ax,
                 label='RMSE', color='mediumblue')
    sns.lineplot(np.arange(len(results_df)), 'MAE', data=results_df, ax=ax,
                 label='MAE', color='Cyan')

    plt.xticks(np.arange(len(results_df)), rotation=45)
    ax.set_xticklabels(results_df['index'])
    ax.set(xlabel="Model",
           ylabel="Scores",
           title="Model Error Comparison")
    sns.despine()

    plt.savefig(f'../model_output/compare_models.png')

def main():
    """Calls functions to compare modelling results"""

    results = create_results_df()
    plot_results(results)

main()
