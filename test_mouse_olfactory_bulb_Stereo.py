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
parser.add_argument("--random-seed", type=int, default=52, help="Random seed for reproducibility")
parser.add_argument("--num-cluster", type=int, default=7, help="Number of clusters to form")
parser.add_argument("--input-expression",type=str, default="/share/home/stu_qilin/software/stdata/5.Mouse_Olfactory/Slide-seqV2_mouse_olfactory_bulb/Puck_200127_15.digital_expression.txt", help="Path to the count file")
parser.add_argument("--input-locations",type=str, default="/share/home/stu_qilin/software/stdata/5.Mouse_Olfactory/Slide-seqV2_mouse_olfactory_bulb/Puck_200127_15_bead_locations.csv", help="Space coordinate file")
parser.add_argument("--input-used-barcode",type=str, default="/share/home/stu_qilin/software/stdata/5.Mouse_Olfactory/Slide-seqV2_mouse_olfactory_bulb/used_barcodes.txt", help="Available barcode")
parser.add_argument("--output-dir", type=str, default="/share/home/stu_qilin/software/DGAST/output_mouse_olfactory_bulb_Slide", help="Directory path for output files")
args = parser.parse_args()
r=args.random_seed
counts_file = args.input_expression
coor_file = args.input_locations
input_used_barcode=args.input_used_barcode
output_dir = args.output_dir
num_cluster=args.num_cluster
counts = pd.read_table(counts_file, sep='\t', index_col=0)
coor_df = pd.read_csv(coor_file, sep='\t')

print(counts.shape, coor_df.shape)
counts.columns = ['Spot_'+str(x) for x in counts.columns]
coor_df.index = coor_df['label'].map(lambda x: 'Spot_'+str(x))
coor_df = coor_df.loc[:, ['x','y']]

adata = sc.AnnData(counts.T)
adata.var_names_make_unique()
coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
adata.obsm["spatial"] = coor_df.to_numpy()
sc.pp.calculate_qc_metrics(adata, inplace=True)


##draw raw
plt.rcParams["figure.figsize"] = (5,4)
sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False,title="Mouse Olfactory n_genes by counts")
plt.axis('off')
plt.savefig(output_dir+"/Mouse_Olfactory_raw.png", dpi=150)
plt.close()

used_barcode = pd.read_csv(input_used_barcode, sep='\t', header=None)
used_barcode = used_barcode[0]
adata = adata[used_barcode,]
###draw filter
plt.rcParams["figure.figsize"] = (5,4)
sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts",s=10, show=False, title='Removing spots outside the main tissue area')
plt.axis('off')
plt.savefig(output_dir+"/Mouse_Olfactory_filter.png", dpi=150)
plt.close()

adata.var["mt"] = adata.var_names.str.startswith("mt-")
adata.var["ercc"] = adata.var_names.str.startswith("ercc-")
adata = adata[:, ~(adata.var["mt"] | adata.var["ercc"])]

sc.pp.filter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=6000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

STMGraph.Cal_Spatial_Net(adata, rad_cutoff=50)
# STAGATEV2.Cal_Spatial_Net(adata, model="KNN",k_cutoff=12)
# STAGATEV2.Stats_Spatial_Net(adata)

adata = STMGraph.train_STMGraph(adata, mask_ratio=0.5,noise=0.05, alpha=1, random_seed=int(r),n_epochs=1500)
adata = STMGraph.mclust_R(adata, used_obsm='STMGraph', num_cluster=7,random_seed=int(r))
# adata.obsm["spatial"] = adata.obsm["spatial"] * (-1)

# adata.obsm['spatial'][:, 1] = -1*adata.obsm['spatial'][:, 1]
plt.rcParams["figure.figsize"] = (5, 5)
sc.pl.embedding(adata, basis="spatial", color="mclust",s=10, show=False, title='STMGraph')
plt.axis('off')

plt.savefig(output_dir+"/Mouse_Olfactory_"+str(r)+".png", dpi=150)
plt.close()

plt.rcParams["figure.figsize"] = (5, 5)
sc.pp.neighbors(adata, use_rep='STMGraph')
sc.tl.umap(adata)
sc.tl.louvain(adata, resolution=0.3)
sc.pl.umap(adata, color='louvain', title='STMGraph')
plt.axis('off')

plt.savefig(output_dir+"/Mouse_Olfactory_"+"louvain"+str(r)+".png", dpi=150)
plt.close()
adata.write_h5ad(output_dir+"/results_output_Mouse_Olfactory"+str(r)+".h5ad")
