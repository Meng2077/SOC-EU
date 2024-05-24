import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def calc_ccc(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    cov_matrix = np.cov(y_true, y_pred)
    covariance = cov_matrix[0, 1]
    var_true = cov_matrix[0, 0]
    var_pred = cov_matrix[1, 1]
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc

def accuracy_plot(y_test, y_pred, title_text, show_range = [0, 6.2]):
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    ccc = calc_ccc(y_test, y_pred)
    
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(title_text, fontsize=20, fontweight='bold')
    plt.title(f'R2={r2:.2f}, RMSE={rmse:.4f}, CCC={ccc:.2f}')
    plt.hexbin(y_test, y_pred, gridsize=(300, 300), cmap='plasma_r', mincnt=1, vmax=30)
    
    plt.xlabel('SOC - test')
    plt.ylabel('SOC - pred')
    
    ax = plt.gca()
    ax.set_aspect('auto', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.plot(show_range, show_range, "-k", alpha=.5)
    
    cax = fig.add_axes([ax.get_position().x1 + 0.05, ax.get_position().y0, 0.02, ax.get_position().height])
    cb = plt.colorbar(cax=cax)
    
    plt.show()