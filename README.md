# STMGraph

## Overview
 ![Image text](https://github.com/binbin-coder/SpatialG/blob/main/overview.jpg)
    Spatial transcriptomics is an emerging single-cell analysis technique that enables high-throughput RNA sequencing on tissue or cell slices to obtain gene expression information of individual cells and map their spatial locations to the tissue structure. Common methods used in spatial transcriptomics research include 10x Genomics Visium, Slide-seq, Stereo-seq, STARmap, DBiT-seq, and Seq-Scope. Interpreting the spatial context of spots within tissues requires careful utilization of their spatial positional information. Although there are already many methods applied to spatial domain detection, with the increase of sequencing platforms and upgrades in methodologies, existing software is no longer able to fully meet the demands. Therefore, we have developed an autoencoder called STMGraph that combines MASK and RE-MASK with dynamic graph attention. By recovering the masked spot information, it captures the latent embedding using a multi-layer perceptron (MLP). The embedding generated from STMGraph not only can we improve the accuracy of spatial domain identification, but we can also regenerate gene expression profiles with reduced noise. The graph theory-based algorithm proposed by this research institute does not rely on the actual measurement of probe-based transcriptome information and does not only calculate the correlation between its own nodes and neighboring nodes. Therefore, it can greatly suppress the information of nodes with poor sequencing quality. We validated the robustness of STMGraph on spatial transcriptomics datasets from different platforms. Furthermore, this algorithm also performs well in removing batch effects in clustering tasks that involve multiple slices.

## Software dependencies
numpy==1.21.5  
r-base==3.5.0  
r-mclust==5.4.6  
rpy2==3.1.0  
tensorflow==1.15.0  
scanpy==1.9.1

## Installation
conda env create -f environment.yaml
