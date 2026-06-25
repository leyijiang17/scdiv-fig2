#Compute Hill numbers for mouse development single-cell atlas

from pathlib import Path
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd

base = Path('/projects/project-ljiang/diversity')

h5ad_path = base / 'mouse_qiu2024.h5ad'
adata = sc.read_h5ad(h5ad_path)

groupby = 'embryo_id'
cell_type_key = 'celltype_update'
major_trajectory_key = 'major_trajectory'
leiden_key = 'leiden'

def celltypes_in_sample(sample, key=cell_type_key):
    # List of cell types in the given sample (drop missing trajectories)
    ct = adata.obs.loc[adata.obs[groupby] == sample, key].dropna()
    # keep original type order-agnostic but remove any non-string artifacts
    return list(ct.astype(str).unique())

def hill_number(dist, q):

    if q == 1:
        return product(x**(-x) for x in dist)

    return sum(x**q for x in dist)**(1/(1-q))

def get_hill_number(sample, q, key=cell_type_key):

    my_celltypes = celltypes_in_sample(sample, key=key)
    if len(my_celltypes) == 0:
        return np.nan

    ct_series = adata.obs.loc[adata.obs[groupby] == sample, key].dropna().astype(str)
    if len(ct_series) == 0:
        return np.nan

    counts = np.unique(ct_series, return_counts=True)
    dist = {ct: 0 for ct in my_celltypes}
    for ct, n in zip(*counts):
        dist[ct] = n

    num_obs = counts[1].sum()
    if num_obs == 0:
        return np.nan

    dist = np.array([dist.get(ct, 0) for ct in my_celltypes]) / num_obs

    return hill_number(dist, q)


q = 2
samples = adata.obs[groupby].unique()

adata.uns["hill_coarse"] = {s: get_hill_number(s, q, key=major_trajectory_key) for s in samples}
adata.uns["hill_fine"] = {s: get_hill_number(s, q, key=cell_type_key) for s in samples}
adata.uns["hill_leiden"] = {s: get_hill_number(s, q, key=leiden_key) for s in samples}
adata.uns["hill_singleton"] = {s: get_hill_number(s, q, key='cell_id') for s in samples}


df_diversity = pd.DataFrame({
    'hill_coarse': adata.uns["hill_coarse"],
    'hill_fine': adata.uns["hill_fine"],
    'hill_leiden': adata.uns["hill_leiden"],
    'hill_singleton': adata.uns["hill_singleton"]
})

group_to_day = adata.obs.groupby(groupby)['day'].first()
df_diversity['day'] = df_diversity.index.map(group_to_day)
df_diversity = df_diversity.reset_index().rename(columns={'index': 'group'})

df_diversity.to_csv(base / 'mouse_hill_results.csv', index=False)
