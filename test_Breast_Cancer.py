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
parser.add_argument("--num-cluster", type=int, default=20, help="Number of clusters to form")
parser.add_argument("--input-ground-true", type=str, default="/share/home/stu_qilin/project/HumanPilot-master/outputs/SpatialDE_clustering/cluster_labels_151673.csv", help="Path to the input ground truth file")
parser.add_argument("--input-dir", type=str, default="/share/home/stu_qilin/project/jupyter/data/test_data/10X/151673", help="Directory path for input data")
parser.add_argument("--count-file", type=str, default="/share/home/stu_qilin/project/jupyter/data/test_data/10X/151673/151673_filtered_feature_bc_matrix.h5", help="Path to the count file")
parser.add_argument("--output-file", type=str, default="/share/home/stu_qilin/software/stgatev2_file/output_151673/", help="Directory path for output files")
args = parser.parse_args()

r=args.random_seed
input_ground_true=args.input_ground_true
input_dir=args.input_dir
count_file=args.count_file
output_file=args.output_file
num_cluster=args.num_cluster
adata = sc.read_visium(path=input_dir, count_file=count_file)
adata.var_names_make_unique()

# Normalization
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ercc"] = adata.var_names.str.startswith("ERCC-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt","ercc"], inplace=True)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

cluser=pd.read_csv(input_ground_true,sep="\t")
adata.obs['Ground Truth']=cluser.loc[:,'ground_truth'].to_list()

STMGraph.Cal_Spatial_Net(adata, k_cutoff=6,model='KNN')
STMGraph.Stats_Spatial_Net(adata)

adata = STMGraph.train_STMGraph(adata, mask_ratio=0.5,alpha= 3.0, random_seed=int(r))

#sc.pp.neighbors(adata, use_rep='STAGATE')
#sc.tl.umap(adata)
adata = STMGraph.mclust_R(adata, used_obsm='STMGraph', num_cluster=num_cluster,random_seed=int(r))

obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
print('Adjusted rand index = %.2f' %ARI,"random_seed =",str(r))

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, color=["mclust", "Ground Truth"], title=['STMGraph (ARI=%.2f)'%ARI, "Ground Truth"])
output_file1=output_file+'/'+input_dir.split('/')[-1]+'ARI_'+str(ARI)+'r_'+str(r)+'.png'
plt.savefig(output_file1, dpi=300)
plt.close()

refine_ARI = adjusted_rand_score(obs_df['refine_mclust'], obs_df['Ground Truth'])
print('Adjusted rand index = %.2f' %refine_ARI,"random_seed =",str(r))

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, color=["refine_mclust", "Ground Truth"], title=['STMGraph (ARI=%.2f)'%refine_ARI, "Ground Truth"])
output_file2=output_file+'/'+'refine_'+input_dir.split('/')[-1]+'ARI_'+str(refine_ARI)+'r_'+str(r)+'.png'
plt.savefig(output_file2, dpi=300)
plt.close()

adata.write_h5ad(output_file+'/'+"results_"+input_dir.split('/')[-1]+'ARI_'+str(refine_ARI)+'r_'+str(r)+".h5ad")
