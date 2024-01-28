import numpy as np
import scipy.sparse as sp
from .STMGraph import STMGraph
from .STDGraph import STDGraph
import tensorflow.compat.v1 as tf
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

def train_STMGraph(adata, hidden_dims=[512,30], mask_ratio=0.5,noise=0.05, n_epochs=1000, lr=0.001, key_added='STMGraph',
                gradient_clipping=5, nonlinear=True, weight_decay=0.0001,verbose=True, alpha=1,
                random_seed=1, pre_labels=None, pre_resolution=0.2,
                save_attention=False, save_loss=False, save_reconstrction=False):
    """
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    alpha
        The weight of cell type-aware spatial neighbor network.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    nonlinear
        If True, the nonlinear avtivation is performed.
    weight_decay
        Weight decay for AdamOptimizer.
    save_attention
        If True, the weights of the attention layers are saved in adata.uns['STAGATE_attention']
    save_loss
        If True, the training loss is saved in adata.uns['STAGATE_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STAGATE_ReX'].

    Returns
    -------
    AnnData
    """

    tf.reset_default_graph()
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    
    if issparse(adata.X):
        X = pd.DataFrame(adata_Vars.X.toarray()[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)
    else:
        X = pd.DataFrame(adata_Vars.X[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)
    #X = pd.DataFrame(adata_Vars.X.toarray()[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)
    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G_tf = prepare_graph_data(G)
    if mask_ratio>0:
        trainer = STMGraph(hidden_dims=[X.shape[1]] + hidden_dims,alpha = alpha,
                    n_epochs=n_epochs, lr=lr, gradient_clipping=gradient_clipping, 
                    nonlinear=nonlinear,weight_decay=weight_decay, verbose=verbose, 
                    random_seed=random_seed)

        trainer(G_tf, X, mask_ratio, noise)
        embeddings, attentions, loss, ReX= trainer.infer(G_tf, X,mask_ratio=0,noise=0)
        
    else:
        trainer = STDGraph(hidden_dims=[X.shape[1]] + hidden_dims,alpha = alpha,
                    n_epochs=n_epochs, lr=lr, gradient_clipping=gradient_clipping,
                    nonlinear=nonlinear,weight_decay=weight_decay, verbose=verbose,
                    random_seed=random_seed)

        trainer(G_tf, X, mask_ratio, noise)
        embeddings, attentions, loss, ReX= trainer.infer(G_tf, X,mask_ratio=0,noise=0)
    cell_reps = pd.DataFrame(embeddings)
    cell_reps.index = cells

    adata.obsm[key_added] = cell_reps.loc[adata.obs_names, ].values
    if save_attention:
        adata.uns['STMGraph_attention'] = attentions
    if save_loss:
        adata.uns['STMGraph_loss'] = loss
    if save_reconstrction:
        ReX = pd.DataFrame(ReX, index=X.index, columns=X.columns)
        ReX[ReX<0] = 0
        adata.layers['STMGraph_ReX'] = ReX.values
    return adata


def prune_spatial_Net(Graph_df, label):
    print('------Pruning the graph...')
    print('%d edges before pruning.' %Graph_df.shape[0])
    pro_labels_dict = dict(zip(list(label.index), label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label']==Graph_df['Cell2_label'],]
    print('%d edges after pruning.' %Graph_df.shape[0])
    return Graph_df


def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)# self-loop
    #data =  adj.tocoo().data
    #adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape)

def recovery_Imputed_Count(adata, size_factor):
    assert('ReX' in adata.uns)
    temp_df = adata.uns['ReX'].copy()
    sf = size_factor.loc[temp_df.index]
    temp_df = np.expm1(temp_df)
    temp_df = (temp_df.T * sf).T
    adata.uns['ReX_Count'] = temp_df
    return adata
