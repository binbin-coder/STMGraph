import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics.cluster import adjusted_rand_score
import STMGraph
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--random-seed", type=int, default=100)
parser.add_argument("--num-cluster", type=int, default=7)
parser.add_argument("--input-file", type=str, default="/share/home/stu_qilin/project/jupyter/data/test_data/10X/151673")
parser.add_argument("--output-file", type=str, default="/share/home/stu_qilin/software/stgatev2_file/output_151673/")
args = parser.parse_args()

r=args.random_seed
input_file=args.input_file
output_file=args.output_file
num_cluster=args.num_cluster
adata = sc.read(filename =input_file)

# Normalization
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ercc"] = adata.var_names.str.startswith("ERCC-")
adata = adata[:, ~(adata.var["mt"] | adata.var["ercc"])]
sc.pp.filter_genes(adata, min_cells=3)
if len(adata.var_names)>3000:
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

STMGraph.Cal_Spatial_Net(adata, rad_cutoff=400)
STMGraph.Stats_Spatial_Net(adata)

adata = STMGraph.train_STMGraph(adata, mask_ratio=0.5,random_seed=int(r),n_epochs=1000)
sc.pp.neighbors(adata, use_rep='STMGraph')
sc.tl.umap(adata)
adata = STMGraph.mclust_R(adata, used_obsm='STMGraph', num_cluster=num_cluster,random_seed=int(r))

ARI = adjusted_rand_score(adata.obs['mclust'], adata.obs['label'])
print('Adjusted rand index = %.2f' %ARI,"random_seed =",str(r))

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, color=["mclust", "label"], title=['STMGraph (ARI=%.2f)'%ARI, "Ground Truth"], spot_size=95)
output_file1=output_file+'/'+input_file.split('/')[-1].split('.h5ad')[0]+'ARI_'+str(ARI)+'r_'+str(r)+'.png'
plt.savefig(output_file1, dpi=300)
plt.close()

refine_ARI = adjusted_rand_score(adata.obs['refine_mclust'], adata.obs['label'])
print('Adjusted rand index = %.2f' %refine_ARI,"random_seed =",str(r))

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, color=["refine_mclust", "label"], title=['STMGraph (ARI=%.2f)'%refine_ARI, "Ground Truth"],spot_size=95)
output_file1=output_file+'/'+input_file.split('/')[-1].split('.h5ad')[0]+'refine_ARI_'+str(refine_ARI)+'r_'+str(r)+'.png'
plt.savefig(output_file1, dpi=300)
plt.close()

adata.write_h5ad(output_file+'/'+"results_"+input_file.split('/')[-1].split('.h5ad')[0]+'ARI_'+str(ARI)+'r_'+str(r)+".h5ad")
