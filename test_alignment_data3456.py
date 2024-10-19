import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics.cluster import adjusted_rand_score
import STMGraph as STMGraph
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--random-seed", type=int, default=52, help="Random seed for reproducibility")
parser.add_argument("--num-cluster", type=int, default=7, help="Number of clusters to form")
parser.add_argument("--k-cutoff", type=int, default=13, help="SNG Maximum number of overwritten neighbors")
parser.add_argument("--alpha", type=int, default=1, help="scaling factors in loss function")
parser.add_argument("--input-h5ad", type=str, default="/share/home/stu_qilin/project/jupyter/paste-test/merge_adata2.h5ad", help="Path to the count file")
parser.add_argument("--output-file", type=str, default="/share/home/stu_qilin/software/STMGraph/output_paste_test", help="Directory path for output files")
args = parser.parse_args()

r=args.random_seed
input_h5ad=args.input_h5ad
output_file=args.output_file
num_cluster=args.num_cluster
k=args.k_cutoff
alpha=args.alpha

adata=sc.read_h5ad(input_h5ad)
adata.var_names_make_unique()

# Normalization
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ercc"] = adata.var_names.str.startswith("ERCC-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt","ercc"], inplace=True)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# STMGraph.Cal_Spatial_Net(adata, rad_cutoff=150)
STMGraph.Cal_Spatial_3DNet(adata, k_cutoff=int(k), model='KNN')
#STMGraph.Stats_Spatial_Net(adata)

adata = STMGraph.train_STMGraph(adata, mask_ratio=0.5, n_epochs=1500, alpha=int(alpha), random_seed=int(r))

# sc.pp.neighbors(adata, use_rep='STAGATE')
# sc.tl.umap(adata)
adata = STMGraph.mclust_R(adata, used_obsm='STMGraph', num_cluster=num_cluster,random_seed=int(r))

sample_list = sorted(list(set(adata.obs['sample'])))
adata_list = []
for sample in sample_list:
    adata_list.append(adata[adata.obs['sample']==sample, :])

fig, ax_list = plt.subplots(1, 4, figsize=(14, 3))
for i,sample in zip(range(4),sample_list):
    sc.pl.embedding(adata_list[i],
               basis='spatial',
               color='sample',
               show = False,
               s=20,
               title=sample,
               ax = ax_list[i])
plt.tight_layout(w_pad=0.2)
output_file0=output_file+'/'+input_h5ad.split('/')[-1].split('.')[0]+'section_alignment'+'.png'
plt.savefig(output_file0, dpi=300)
plt.close()

fig, ax_list = plt.subplots(1, 4, figsize=(12, 3))
SUM_reARI=0
for i,sample in zip(range(4),sample_list):
    # filter out NA nodes
    adata_a = adata_list[i][~pd.isnull(adata_list[i].obs['Ground Truth'])]
    # calculate metric ARI
    ARI = adjusted_rand_score(adata_a.obs['refine_mclust'], adata_a.obs['Ground Truth'])
    SUM_reARI+=ARI
    sc.pl.embedding(adata_list[i],
               basis='spatial',
               color='refine_mclust',
               show = False,
               s=25,
               title=sample+'_ARI=%.8f'%ARI,
               ax = ax_list[i])
plt.tight_layout(w_pad=0.2)
output_file1=output_file+'/'+input_h5ad.split('/')[-1].split('.')[0]+'reARI_'+str(SUM_reARI/4)+'r_'+str(r)+'k_'+str(k)+'alpha_'+str(alpha)+'.png'
plt.savefig(output_file1, dpi=300)
plt.close()

fig, ax_list = plt.subplots(1, 4, figsize=(12, 3))
SUM_ARI=0
for i,sample in zip(range(4),sample_list):
    # filter out NA nodes
    adata_a = adata_list[i][~pd.isnull(adata_list[i].obs['Ground Truth'])]
    # calculate metric ARI
    ARI = adjusted_rand_score(adata_a.obs['mclust'], adata_a.obs['Ground Truth'])
    SUM_ARI+=ARI
    sc.pl.embedding(adata_list[i],
               basis='spatial',
               color='mclust',
               show = False,
               s=25,
               title=sample+'_ARI=%.8f'%ARI,
               ax = ax_list[i])
plt.tight_layout(w_pad=0.2)
output_file2=output_file+'/'+input_h5ad.split('/')[-1].split('.')[0]+'ARI_'+str(SUM_ARI/4)+'r_'+str(r)+'k_'+str(k)+'alpha_'+str(alpha)+'.png'
plt.savefig(output_file2, dpi=300)
plt.close()

adata.write_h5ad(output_file+'/'+"results_"+input_h5ad.split('/')[-1]+'ARI_'+str(SUM_reARI/4)+'r_'+str(r)+'k_'+str(k)+'alpha_'+str(alpha)+".h5ad")
