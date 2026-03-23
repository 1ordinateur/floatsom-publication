import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def topology_pca_one(df, measures=['topographic_error', 'neighborhood_preservation', 'topographic_function'], standardize=True):
    """
    Combine topology measures with a PCA
    df: pandas DataFrame with the measures
    standardize: boolean to standardize the data before PCA

    Returns:
    pc1_scores: numpy array of the first principal component scores
    pca_info: dictionary with PCA information
    """
    # Select the measures
    df_four_measures = df[measures].values

    # Standardize the data
    if standardize:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_four_measures)
    else:
        df_scaled = df_four_measures
    
    # Perform PCA
    pca = PCA()
    pca_scores = pca.fit_transform(df_scaled)

    # Extract first PC
    pc1_scores = pca_scores[:, 0]

    # Info Dictionary
    pca_info = {
        'pca_object': pca,
        'scaler': scaler if standardize else None,
        'explained_variance': pca.explained_variance_ratio_,
        'loadings': pca.components_[0],
        'measures': measures,
        'pc1_scores': pc1_scores,
        'pca_scores': pca_scores,
        'df_scaled': df_scaled,
        'df_four_measures': df_four_measures
    }

    return pc1_scores, pca_info


def add_pc1_to_df(df, measures=['topographic_error', 'neighborhood_preservation', 'topographic_function'], column_name='Topology_PC1', standardize=True):
    """
    Adds the first principal component (PC1) from a PCA of the specified measures to the DataFrame as a new column.

    Parameters:
    df: pandas DataFrame containing the measures.
    measures: list of strings, column names to use for PCA (default: all four topology measures).
    column_name: string, name of the new column to store PC1 scores (default: 'Topology_PC1').
    standardize: boolean, whether to standardize the measures before PCA (default: True).

    Returns:
    df_new: DataFrame with the new PC1 column added.
    pca_info: dictionary with PCA details (see topology_pca_one for keys).
    """

    df_new = df.copy()

    pc1_scores, pca_info = topology_pca_one(df_new, measures, standardize)

    df_new[column_name] = pc1_scores

    return df_new, pca_info
