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
parser.add_argument("--random-seed", type=int, default=52)
parser.add_argument("--input-expression",type=str, default="/share/home/stu_qilin/software/stdata/6.Mouse_Hippocampus_Tissue/Puck_200115_08.digital_expression.txt")
parser.add_argument("--input-locations",type=str, default="/share/home/stu_qilin/software/stdata/6.Mouse_Hippocampus_Tissue/Puck_200115_08_bead_locations.csv")
parser.add_argument("--input-used-barcode",type=str, default="/share/home/stu_qilin/software/stdata/6.Mouse_Hippocampus_Tissue/used_barcodes.csv")
parser.add_argument("--output-dir", type=str, default="/share/home/stu_qilin/software/STMGraph7/output_mouse_Hippocampus_Slide")
args = parser.parse_args()
r=args.random_seed
counts_file = args.input_expression
coor_file = args.input_locations
input_used_barcode=args.input_used_barcode
output_dir = args.output_dir
counts = pd.read_table(counts_file, sep='\t', index_col=0)
coor_df = pd.read_csv(coor_file, sep=',',index_col=0)
# coor_df = coor_df.set_index(coor_df.columns[2])

print(counts.shape, coor_df.shape)
adata = sc.AnnData(counts.T)
adata.var_names_make_unique()
coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
adata.obsm["spatial"] = coor_df.to_numpy()

sc.pp.calculate_qc_metrics(adata, inplace=True)
##draw raw
plt.rcParams["figure.figsize"] = (6,5.5)
sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts",s=6, show=False, title='mouse hippocampus')
plt.axis('off')
plt.savefig(output_dir+"/Mouse_hippocampus_raw.png", dpi=150)
plt.close()

used_barcode = pd.read_csv(input_used_barcode, sep=',', index_col=0, header=0)
used_barcode = used_barcode.iloc[:,0]
adata = adata[used_barcode,]
###draw filter
plt.rcParams["figure.figsize"] = (6,5.5)
sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts",s=10, show=False, title='Removing spots outside the main tissue area')
plt.axis('off')
plt.savefig(output_dir+"/Mouse_hippocampus_filter.png", dpi=150)
plt.close()

adata.var["mt"] = adata.var_names.str.startswith("mt-")
adata.var["ercc"] = adata.var_names.str.startswith("ercc-")
adata = adata[:, ~(adata.var["mt"] | adata.var["ercc"])]#删除线粒体和外源基因

sc.pp.filter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

STMGraph.Cal_Spatial_Net(adata, rad_cutoff=50)
# STAGATEV2.Cal_Spatial_Net(adata, model="KNN",k_cutoff=12)
# STAGATEV2.Stats_Spatial_Net(adata)

adata = STMGraph.train_STMGraph(adata, mask_ratio=0.5,noise=0.05,alpha=1,random_seed=int(r),n_epochs=2000)
adata = STMGraph.mclust_R(adata, used_obsm='STMGraph', num_cluster=10,random_seed=int(r))
# adata.obsm["spatial"] = adata.obsm["spatial"] * (-1)

# adata.obsm['spatial'][:, 1] = -1*adata.obsm['spatial'][:, 1]
plt.rcParams["figure.figsize"] = (5, 5)
sc.pl.embedding(adata, basis="spatial", color="mclust",s=3, show=False, title='STMGraph')
plt.axis('off')

plt.savefig(output_dir+"/Mouse_hippocampus_"+str(r)+".png", dpi=150)
plt.close()

plt.rcParams["figure.figsize"] = (5, 5)
sc.pp.neighbors(adata, use_rep='STMGraph')
sc.tl.umap(adata)
sc.tl.louvain(adata, resolution=0.3)
sc.pl.umap(adata, color='louvain', title='STMGraph')
plt.axis('off')

plt.savefig(output_dir+"/Mouse_hippocampus_"+"louvain"+str(r)+".png", dpi=150)
plt.close()
adata.write_h5ad(output_dir+"/results_output_Mouse_hippocampus"+str(r)+".h5ad")
