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
    plt.hexbin(y_pred, y_test, gridsize=(300, 300), cmap='plasma_r', mincnt=1, vmax=30)
    
    plt.xlabel('SOC - pred')
    plt.ylabel('SOC - true')
    
    ax = plt.gca()
    ax.set_aspect('auto', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.plot(show_range, show_range, "-k", alpha=.5)
    
    cax = fig.add_axes([ax.get_position().x1 + 0.05, ax.get_position().y0, 0.02, ax.get_position().height])
    cb = plt.colorbar(cax=cax)
    
    plt.show()
    
# def clean_prop(df, prop, limit):
#     print(f'\033[1mCleaning {prop}\033[0m')
#     tot = len(df)
#     print(f'originally with {tot} rows')
#     # Clean NaN
#     num = df[prop].isna().sum()
#     ccol = df.loc[df[prop].isna()]['ref'].unique()
#     print(f'{num} ({num/tot*100:.2f}%) rows with NaN, from {ccol}')
#     df = df.dropna(subset=[prop])
    
#     # check if there are string values that cannot be converted to numerical values,
#     # usually it's <LOD (limit of detection), such as '<6', '<LOD', etc
# #     df.loc[:,prop] = pd.to_numeric(df.loc[:,prop], errors='coerce')
#     df[prop] = pd.to_numeric(df[prop], errors='coerce')
#     num = df[prop].isna().sum()
#     ccol = df.loc[df[prop].isna()]['ref'].unique()
#     print(f'{num} ({num/tot*100:.2f}%) rows with invalid strings, from {ccol}')
#     df = df.dropna(subset=[prop])
    
#     # Check for values below 0, which are invalid for all properties
#     num = len(df.loc[df[prop] < 0])
#     ccol = df.loc[df[prop] < 0]['ref'].unique()
#     print(f'{num} ({num/tot*100:.2f}%) rows with {prop} < 0, from {ccol}')
#     df = df[df[prop] >= 0]
    
#     # check for values higher than plausible limit
#     if limit:
#         num = len(df.loc[df[prop]>limit])
#         ccol = df.loc[df[prop]>limit]['ref'].unique()
#         print(f'{num} ({num/tot*100:.2f}%) rows with {prop} > limit values, from {ccol}')
#         df = df[df[prop] < limit]
    
#     print(f'{len(df)} valid data records left')
#     return df


# dff = clean_prop(df,'oc',1000)