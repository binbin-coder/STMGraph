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
parser.add_argument("--random-seed", type=int, default=1, help="Random seed for reproducibility")
parser.add_argument("--num-cluster", type=int, default=7, help="Number of clusters to form")
parser.add_argument("--k-cutoff", type=int, default=8, help="SNG Maximum number of overwritten neighbors")
parser.add_argument("--input-file", type=str, default="/share/home/stu_qilin/software/stdata/Mouse_embryo/E9.5_E1S1.MOSTA.h5ad", help="Directory path for input data")
parser.add_argument("--output-file", type=str, default="/share/home/stu_qilin/software/STMGraph/output_Mouse_embryo9.5/", help="Directory path for output files")
args = parser.parse_args()

r=args.random_seed
input_file=args.input_file
output_file=args.output_file
num_cluster=args.num_cluster
k=args.k_cutoff
adata = sc.read(filename =input_file)

# Normalization
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ercc"] = adata.var_names.str.startswith("ERCC-")
adata = adata[:, ~(adata.var["mt"] | adata.var["ercc"])]
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

STMGraph.Cal_Spatial_Net(adata, k_cutoff=int(k), model='KNN', delta_err=0.4)
STMGraph.Stats_Spatial_Net(adata)

adata = STMGraph.train_STMGraph(adata, mask_ratio=0.5,alpha=1,random_seed=int(r))

sc.pp.neighbors(adata, use_rep='STMGraph')
sc.tl.umap(adata)
adata = STMGraph.mclust_R(adata, used_obsm='STMGraph', num_cluster=num_cluster,random_seed=int(r))

ARI = adjusted_rand_score(adata.obs['mclust'], adata.obs['annotation'])
print('Adjusted rand index = %.2f' %ARI,"random_seed =",str(r))
adata.obsm['spatial'][:, 1] = -1*adata.obsm['spatial'][:, 1]

plt.rcParams["figure.figsize"] = (6, 8)
sc.pl.embedding(adata, basis="spatial", color="mclust",s=50, show=False, title='STMGraph ARI =%.2f' %ARI)
output_file1=output_file+'/'+input_file.split('/')[-1].split('.h5ad')[0]+'ARI_'+str(ARI)+'r_'+str(r)+'.png'
plt.savefig(output_file1, dpi=100)
plt.close()

refine_ARI = adjusted_rand_score(adata.obs['refine_mclust'], adata.obs['annotation'])
print('Adjusted rand index = %.2f' %refine_ARI,"random_seed =",str(r))

plt.rcParams["figure.figsize"] = (6, 8)
sc.pl.embedding(adata, basis="spatial", color="refine_mclust",s=50, show=False, title='STMGraph ARI =%.2f' %refine_ARI)
output_file1=output_file+'/'+input_file.split('/')[-1].split('.h5ad')[0]+'refine_ARI_'+str(refine_ARI)+'r_'+str(r)+'.png'
plt.savefig(output_file1, dpi=100)
plt.close()

plt.rcParams["figure.figsize"] = (6, 8)
sc.pl.embedding(adata, basis="spatial", color="annotation",s=50, show=False, title='Ground Truth')
output_file1=output_file+'/'+input_file.split('/')[-1].split('.h5ad')[0]+'annotation.png'
plt.savefig(output_file1, dpi=100)
plt.close()

adata.write_h5ad(output_file+'/'+"results_"+input_file.split('/')[-1].split('.h5ad')[0]+'ARI_'+str(ARI)+'r_'+str(r)+".h5ad")
