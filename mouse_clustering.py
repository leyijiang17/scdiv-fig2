"""
Seurat-style clustering pipeline for large h5ad datasets.
Steps: incremental PCA → KNN graph → Leiden → UMAP → save.

Seurat equivalents:
  NormalizeData(scale.factor=1e4) + log1p   → normalise_batch()
  RunPCA(npcs=N_COMPS)                       → IncrementalPCA
  FindNeighbors(dims=1:N_COMPS, k=20)        → sc.pp.neighbors(n_neighbors=20)
  FindClusters(resolution=0.5)               → sc.tl.leiden(resolution=0.5)
  RunUMAP(dims=1:N_COMPS)                    → sc.tl.umap()

Usage:
  python run_clustering.py \\
      --h5ad /path/to/mouse_all_HVG.h5ad \\
      --outdir /path/to/output \\
      [--n_comps 50] [--chunk 50000] [--n_neighbors 20] [--resolution 0.5]
"""

# ── Thread limits ─────────────────────────────────────────────────────────────
import os
os.environ['OMP_NUM_THREADS']        = '1'
os.environ['OPENBLAS_NUM_THREADS']   = '1'
os.environ['MKL_NUM_THREADS']        = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS']    = '1'

import gc

import argparse
import time
import numpy as np
import h5py
import joblib
import anndata as ad
import scanpy as sc
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.decomposition import IncrementalPCA

# ── CLI arguments ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Seurat-style clustering pipeline')
parser.add_argument('--h5ad',        required=True,               help='Path to input h5ad file')
parser.add_argument('--outdir',      default='.',                  help='Output directory')
parser.add_argument('--n_comps',     type=int,   default=50,       help='PCA components')
parser.add_argument('--chunk',       type=int,   default=50_000,   help='Cells per PCA batch')
parser.add_argument('--n_neighbors', type=int,   default=20,       help='KNN k (Seurat default: 20)')
parser.add_argument('--resolution',  type=float, default=0.5,      help='Leiden resolution (Seurat default: 0.5)')
parser.add_argument('--ckpt_every',  type=int,   default=20,       help='Checkpoint every N batches')
args = parser.parse_args()

H5AD_PATH  = Path(args.h5ad)
OUT_DIR    = Path(args.outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_IPCA  = OUT_DIR / 'ipca_checkpoint.joblib'
CKPT_PCA   = OUT_DIR / 'X_pca.npy'
CKPT_IDX   = OUT_DIR / 'pca_resume_idx.npy'
OUT_H5AD   = OUT_DIR / 'mouse_clustered.h5ad'

N_COMPS    = args.n_comps
CHUNK      = args.chunk
CKPT_EVERY = args.ckpt_every

print(f'h5ad      : {H5AD_PATH}')
print(f'outdir    : {OUT_DIR}')
print(f'n_comps   : {N_COMPS}')
print(f'chunk     : {CHUNK:,}')
print(f'n_neighbors: {args.n_neighbors}')
print(f'resolution: {args.resolution}')


# ── Helper: read CSR rows directly via h5py ────────────────────────────────────
def read_rows(h5f, start, end, n_genes):
    """Read rows [start, end) from h5ad CSR X cleanly, avoiding native 
    h5py array-slicing segfaults on massive datasets.
    """
    ip_start_raw = h5f['X/indptr'][start]
    ip_end_raw = h5f['X/indptr'][end]
    
    if h5f['X/indptr'].dtype == np.int32:
        c0 = int(np.uint32(ip_start_raw))
        c1 = int(np.uint32(ip_end_raw))
    else:
        c0 = int(ip_start_raw)
        c1 = int(ip_end_raw)
        
    d  = h5f['X/data'][c0:c1].astype(np.float32)
    ix = h5f['X/indices'][c0:c1]
    
    ip_chunk = h5f['X/indptr'][start:end + 1]
    if h5f['X/indptr'].dtype == np.int32:
        ip = ip_chunk.view(np.uint32).astype(np.int64) - c0
    else:
        ip = ip_chunk.astype(np.int64) - c0
    
    return csr_matrix((d, ix, ip), shape=(end - start, n_genes))

def normalise_batch(X_b):
    """10k library-size normalisation + log1p, in-place."""
    s = X_b.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    X_b *= (1e4 / s)
    np.log1p(X_b, out=X_b)
    return X_b


# ── 0. Dataset info ────────────────────────────────────────────────────────────
print('\n=== Dataset info ===')
with h5py.File(H5AD_PATH, 'r') as f:
    n_obs   = int(f['X'].attrs['shape'][0])
    n_genes = int(f['X'].attrs['shape'][1])
    indptr_dtype = f['X/indptr'].dtype
    last_ip = f['X/indptr'][-1]
    if indptr_dtype == np.int32:
        total_nnz = int(last_ip.view(np.uint32))
    else:
        total_nnz = int(last_ip)

print(f'Cells     : {n_obs:,}')
print(f'Genes     : {n_genes:,}')
print(f'Total nnz : {total_nnz:,}  (int32 max = {2**31-1:,})')
print(f'indptr    : {indptr_dtype}  {"⚠ will apply uint32 fix" if indptr_dtype == np.int32 else "OK"}')


# ── 1. Incremental PCA ─────────────────────────────────────────────────────────
print('\n=== Incremental PCA ===')

if CKPT_IDX.exists():
    resume_start, resume_pass = np.load(CKPT_IDX).astype(int)
    ipca  = joblib.load(CKPT_IPCA)
    X_pca = np.load(CKPT_PCA) if CKPT_PCA.exists() else None
    print(f'Resuming from pass {resume_pass}, row {resume_start:,}')
else:
    resume_start, resume_pass = 0, 1
    ipca  = IncrementalPCA(n_components=N_COMPS)
    X_pca = None
    print('Starting fresh.')

# Pass 1 — fit
if resume_pass == 1:
    print(f'Pass 1/2 — fitting from row {resume_start:,} / {n_obs:,}')
    t0 = time.time()
    with h5py.File(H5AD_PATH, 'r') as f:
        for i, start in enumerate(range(resume_start, n_obs, CHUNK)):
            end = min(start + CHUNK, n_obs)
            X_b = read_rows(f, start, end, n_genes).toarray()
            normalise_batch(X_b)
            ipca.partial_fit(X_b)
            del X_b
            gc.collect()
            print(f'  Pass 1: {end/n_obs*100:5.1f}%  ({end:,}/{n_obs:,})', flush=True)
            if (i + 1) % CKPT_EVERY == 0:
                joblib.dump(ipca, CKPT_IPCA)
                np.save(CKPT_IDX, np.array([end, 1]))

    print(f'Pass 1 done in {(time.time()-t0)/60:.1f} min.')
    print(f'Explained variance (top 10): {ipca.explained_variance_ratio_[:10].round(4)}')
    joblib.dump(ipca, CKPT_IPCA)
    np.save(CKPT_IDX, np.array([0, 2]))
    resume_start = 0

# Pass 2 — transform
print(f'Pass 2/2 — transforming from row {resume_start:,} / {n_obs:,}')
if X_pca is None or X_pca.shape != (n_obs, N_COMPS):
    X_pca = np.zeros((n_obs, N_COMPS), dtype=np.float32)

t1 = time.time()
with h5py.File(H5AD_PATH, 'r') as f:
    for i, start in enumerate(range(resume_start, n_obs, CHUNK)):
        end = min(start + CHUNK, n_obs)
        X_b = read_rows(f, start, end, n_genes).toarray()
        normalise_batch(X_b)
        X_pca[start:end] = ipca.transform(X_b)
        del X_b
        gc.collect()
        print(f'  Pass 2: {end/n_obs*100:5.1f}%  ({end:,}/{n_obs:,})', flush=True)
        if (i + 1) % CKPT_EVERY == 0:
            np.save(CKPT_PCA, X_pca)
            np.save(CKPT_IDX, np.array([end, 2]))

np.save(CKPT_PCA, X_pca)
np.save(CKPT_IDX, np.array([n_obs, 2]))   # sentinel: pass 2 fully done
print(f'Pass 2 done in {(time.time()-t1)/60:.1f} min.')
print(f'X_pca: {X_pca.shape},  {X_pca.nbytes / 1e9:.2f} GB')


# ── 2. Build AnnData (metadata only, no X in RAM) ─────────────────────────────
print('\n=== Building AnnData ===')
adata_full = ad.read_h5ad(H5AD_PATH, backed='r')
adata = ad.AnnData(
    obs=adata_full.obs.copy(),
    var=adata_full.var.copy(),
)
adata_full.file.close()
adata.obsm['X_pca'] = X_pca
print(adata)


# ── 3. KNN graph — Seurat: FindNeighbors(dims=1:N_COMPS, k.param=20) ──────────
print(f'\n=== KNN graph (n_neighbors={args.n_neighbors}) ===')
t2 = time.time()
sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=N_COMPS, use_rep='X_pca')
print(f'KNN done in {(time.time()-t2)/60:.1f} min.')


# ── 4. Leiden — Seurat: FindClusters(resolution=...) ─────────────────────────
print(f'\n=== Leiden clustering (resolution={args.resolution}) ===')
t3 = time.time()
sc.tl.leiden(adata, resolution=args.resolution, key_added='leiden',
             flavor='igraph', n_iterations=2, directed=False)
n_clusters = adata.obs['leiden'].nunique()
print(f'Leiden done in {(time.time()-t3)/60:.1f} min.  Clusters: {n_clusters}')
print(adata.obs['leiden'].value_counts().head(20).to_string())


# ── 5. UMAP — Seurat: RunUMAP(dims=1:N_COMPS) ─────────────────────────────────
print(f'\n=== UMAP ===')
t4 = time.time()
sc.tl.umap(adata)
print(f'UMAP done in {(time.time()-t4)/60:.1f} min.')


# ── 6. Save ────────────────────────────────────────────────────────────────────
print(f'\n=== Saving to {OUT_H5AD} ===')
adata.write_h5ad(OUT_H5AD)
print('Done.')
print(f'\nOutput columns in obs: {list(adata.obs.columns)}')
print(f'obsm keys: {list(adata.obsm.keys())}')