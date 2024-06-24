import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
import STMGraph as STMGraph
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--random-seed", type=int, default=52)
parser.add_argument("--input-dir", type=str, default="/share/home/stu_qilin/software/stdata/Mouse_Brain_Section_1")
parser.add_argument("--count-file", type=str, default="/share/home/stu_qilin/software/stdata/Mouse_Brain_Section_1/V1_Adult_Mouse_Brain_Coronal_Section_1_filtered_feature_bc_matrix.h5")
parser.add_argument("--output-file", type=str, default="/share/home/stu_qilin/software/STMGraph/Mouse_Brain_Section_1/")
args = parser.parse_args()

r=args.random_seed
input_dir=args.input_dir
count_file=args.count_file
output_file=args.output_file
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

STMGraph.Cal_Spatial_Net(adata,  k_cutoff=6, model='KNN')
STMGraph.Stats_Spatial_Net(adata)
adata = STMGraph.train_STMGraph(adata, mask_ratio=0.5,noise=0.05 alpha=1, random_seed=int(r), save_attention=True)

import matplotlib as mpl
import networkx as nx
att_df = pd.DataFrame(adata.uns['STMGraph_attention'][0].toarray(), index=adata.obs_names, columns=adata.obs_names)
att_df = att_df.values
for it in range(att_df.shape[0]):
    att_df[it, it] = 0
    
G_atten = nx.from_numpy_matrix(att_df)
M = G_atten.number_of_edges()
edge_colors = range(2, M + 2)
coor_df = pd.DataFrame(adata.obsm['spatial'].copy(), index=adata.obs_names)
coor_df[1] = -1 * coor_df[1]
image_pos = dict(zip(range(coor_df.shape[0]), [np.array(coor_df.iloc[it,]) for it in range(coor_df.shape[0])]))
labels = nx.get_edge_attributes(G_atten,'weight')

fig, ax = plt.subplots(figsize=[13,10])
nx.draw_networkx_nodes(G_atten, image_pos, node_size=5, ax=ax)
cmap = plt.cm.plasma
edges = nx.draw_networkx_edges(G_atten, image_pos, edge_color=labels.values(),width=4, ax=ax,
                               edge_cmap=cmap,edge_vmax=0.25,edge_vmin=0.05)
ax = plt.gca()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = 0.05, vmax=0.25))
sm._A = []
plt.colorbar(sm)

ax.set_axis_off()
output_file2=output_file+'/'+input_dir.split('/')[-1]+'microenvironmental_heterogeneity'+'.png'
plt.savefig(output_file2, dpi=300)
plt.close()

adata.write_h5ad(output_file+'/'+"results_"+input_dir.split('/')[-1]+'microenvironmental_heterogeneity'+".h5ad")
