#Compute LC diversity for mouse development single-cell atlas

from pathlib import Path
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import scdiv

base = Path('/projects/project-ljiang/diversity')

h5ad_path = base / 'mouse_qiu2024.h5ad'
adata = sc.read_h5ad(h5ad_path)

groupby = 'embryo_id'
cell_type_key = 'celltype_update'
major_trajectory_key = 'major_trajectory'
leiden_key = 'leiden'

scdiv.tl.diversity(adata, order=2, alpha=0.5, cell_type_key=major_trajectory_key, groupby=groupby, use_highly_variable=False, key_added='scdiv_coarse')
scdiv.tl.diversity(adata, order=2, alpha=0.5, cell_type_key=cell_type_key, groupby=groupby, use_highly_variable=False, key_added='scdiv_fine')
scdiv.tl.diversity(adata, order=2, alpha=0.5, cell_type_key=leiden_key, groupby=groupby, use_highly_variable=False, key_added='scdiv_leiden')
scdiv.tl.diversity(adata, order=2, alpha=0.5, groupby=groupby, use_highly_variable=False, key_added='scdiv_singleton')

df_diversity = pd.DataFrame({
    'scdiv_coarse': adata.uns["scdiv_coarse"],
    'scdiv_fine': adata.uns["scdiv_fine"],
    'scdiv_leiden': adata.uns["scdiv_leiden"],
    'scdiv_singleton': adata.uns["scdiv_singleton"]
})

group_to_day = adata.obs.groupby(groupby)['day'].first()
df_diversity['day'] = df_diversity.index.map(group_to_day)
df_diversity = df_diversity.reset_index().rename(columns={'index': 'group'})

df_diversity.to_csv(base / 'mouse_scdiv_results.csv', index=False)
