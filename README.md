# STMGraph

## Overview
 ![Image text](https://github.com/binbin-coder/SpatialG/blob/main/overview.jpg)
    Rapid growth of Spatial transcriptomics (ST) technologies enables dissecting the tissue architecture in spatial context. However, the current ST context integration ignoring ST dropouts, generating low signal-to-noise ratio for latent embedding, resulting in challenges in the accuracy and robustness of microenvironmental heterogeneity detecting, spatial domain clustering, and batch-effects correction. Here, we developed an STMGraph, an accurate and universal dual-view deep learning framework that combines dual-remask (MASK-REMASK) with dynamic graph attention model (DGAT) to exploit ST data outperforming pre-existing tools. The dual-remask mechanism reconstructs unmasked nodes and masked nodes separately, guiding features between the two views mutually. DGAT leverages self-supervision to update graph linkage relationships from two distinct perspectives, thereby generating a comprehensive representation for each node. Systematic benchmarking against with ten state-of-the-art tools, reveals that the STMGraph has the best performance with high accuracy and robustness on spatial domain clustering for the datasets of diverse ST platforms from multi- to sub-cellular resolutions. Furthermore, STMGraph aggregates ST information cross regions by dual-remask to realize the batch-effects correction implicitly, allowing for spatial domain clustering of ST multi-slice. STMGraph is platform independent, and superior in spatial-context-aware to achieve microenvironmental heterogeneity detecting, spatial domain clustering, and batch-effects correction , therefore, a desirable novel tool for diverse ST studies.

## Software dependencies
numpy==1.21.5  
r-base==3.5.0  
r-mclust==5.4.6  
rpy2==3.1.0  
tensorflow==1.15.0  
scanpy==1.9.1

## Installation
conda env create -f environment.yaml  
pip install STMGraph

## spatial domain clustering
python test_cluster.py
## batch-effects correction
python test_alignment_data3456.py
## microenvironmental heterogeneity detecting
python microenvironmental_heterogeneity.py
## gene denoising
python test_denoising.py
