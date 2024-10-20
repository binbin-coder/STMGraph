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
parser.add_argument("--k-cutoff", type=int, default=6, help="SNG Maximum number of overwritten neighbors")
parser.add_argument("--input-dir", type=str, default="/share/home/stu_qilin/project/jupyter/data/test_data/10X/151674", help="Directory path for input data")
parser.add_argument("--count-file", type=str, default="/share/home/stu_qilin/project/jupyter/data/test_data/10X/151674/151674_filtered_feature_bc_matrix.h5", help="Path to the count file")
parser.add_argument("--output-file", type=str, default="/share/home/stu_qilin/software/STMGraph/output_151674/", help="Directory path for output files")
args = parser.parse_args()

r=args.random_seed
input_dir=args.input_dir
count_file=args.count_file
output_file=args.output_file
k=args.k_cutoff
adata = sc.read_visium(path=input_dir, count_file=count_file)
adata.var_names_make_unique()

# Normalization
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ercc"] = adata.var_names.str.startswith("ERCC-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt","ercc"], inplace=True)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

STMGraph.Cal_Spatial_Net(adata,  k_cutoff=int(k), model='KNN')
STMGraph.Stats_Spatial_Net(adata)
adata = STMGraph.train_STMGraph(adata, mask_ratio=0.5,alpha=1, random_seed=int(r))

plot_gene = 'RASGRF2'
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[0], title='RAW_'+plot_gene, vmax='p99')
sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[1], title='STMGraph_'+plot_gene, layer='STMGraph_ReX', vmax='p99')
output_file1=output_file+'/'+input_dir.split('/')[-1]+'denoising'+'plot_gene'+'.png'
plt.savefig(output_file1, dpi=300)
plt.close()

plot_gene = 'B3GALT2'
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[0], title='RAW_'+plot_gene, vmax='p99')
sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[1], title='STMGraph_'+plot_gene, layer='STMGraph_ReX', vmax='p99')
output_file2=output_file+'/'+input_dir.split('/')[-1]+'denoising'+'plot_gene'+'.png'
plt.savefig(output_file2, dpi=300)
plt.close()

adata.write_h5ad(output_file+'/'+"results_"+input_dir.split('/')[-1]+'denoising'+".h5ad")
