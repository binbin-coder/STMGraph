import pandas as pd
import numpy as np
import sklearn.neighbors


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True,delta_err=1):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        distance_threshold=np.sort(distances[:,-1])[0]
        distance_threshold = distance_threshold+delta_err
        for it in range(indices.shape[0]):
            close_indices = indices[it, distances[it, :] <= distance_threshold]
            close_distances = distances[it, distances[it, :] <= distance_threshold]
            KNN_list.append(pd.DataFrame(zip([it]*len(close_indices), close_indices, close_distances)))
            # KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

def group_in_order(sequence):
    partitions = []
    start_index = 0
    
    while start_index<len(sequence)-1:
        partition = sequence[start_index:start_index+2]
        partitions.append(partition)
        start_index += 1
    return partitions

def Cal_Spatial_3DNet(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True,delta_err=1):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    assert 'data' in adata.obs or 'sample' in adata.obs, "Error: The 'data' or 'sample' column does not exist in adata.obs."
    coor['sample'] = adata.obs.get('sample', default=adata.obs.get('data'))
    sequence = sorted(list(set(coor['sample'])))
    Spatial_Net_all = []
    for i in group_in_order(sequence):
        filiter_coor = coor[coor['sample'].isin(i)]
        KNN_list = []
        if model == 'Radius':
            nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(filiter_coor.iloc[:,:-1])
            distances, indices = nbrs.radius_neighbors(filiter_coor.iloc[:,:-1], return_distance=True)
            for it in range(indices.shape[0]):
                KNN_list.append(pd.DataFrame(zip(it, x, distances[it][x])))

        if model == 'KNN':
            nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(filiter_coor.iloc[:,:-1])
            distances, indices = nbrs.kneighbors(filiter_coor.iloc[:,:-1])
            distance_threshold=np.sort(distances[:,-1])[0]
            distance_threshold = distance_threshold+delta_err
            for it in range(indices.shape[0]):
                close_indices = indices[it, distances[it, :] <= distance_threshold]
                close_distances = distances[it, distances[it, :] <= distance_threshold]
                KNN_list.append(pd.DataFrame(zip([it]*len(close_indices), close_indices, close_distances)))
    
        KNN_df = pd.concat(KNN_list)

        KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
        Spatial_Net = KNN_df.copy()
        Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
        id_cell_trans = dict(zip(range(filiter_coor.shape[0]), np.array(filiter_coor.index), ))
        Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
        Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
        
        Spatial_Net_all.append(Spatial_Net)
    Spatial_Net_all_df = pd.concat(Spatial_Net_all)
    cleaned_KNN_df = Spatial_Net_all_df.drop_duplicates()
    Spatial_Net = cleaned_KNN_df.copy()
    if verbose:
            print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
            print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net


def Cal_Spatial_Net_3D(adata, rad_cutoff_2D, rad_cutoff_Zaxis,
                       key_section='Section_id', section_order=None, verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff_2D
        radius cutoff for 2D SNN construction.
    rad_cutoff_Zaxis
        radius cutoff for 2D SNN construction for consturcting SNNs between adjacent sections.
    key_section
        The columns names of section_ID in adata.obs.
    section_order
        The order of sections. The SNNs between adjacent sections are constructed according to this order.
    
    Returns
    -------
    The 3D spatial networks are saved in adata.uns['Spatial_Net'].
    """
    adata.uns['Spatial_Net_2D'] = pd.DataFrame()
    adata.uns['Spatial_Net_Zaxis'] = pd.DataFrame()
    num_section = np.unique(adata.obs[key_section]).shape[0]
    if verbose:
        print('Radius used for 2D SNN:', rad_cutoff_2D)
        print('Radius used for SNN between sections:', rad_cutoff_Zaxis)
    for temp_section in np.unique(adata.obs[key_section]):
        if verbose:
            print('------Calculating 2D SNN of section ', temp_section)
        temp_adata = adata[adata.obs[key_section] == temp_section, ]
        Cal_Spatial_Net(
            temp_adata, rad_cutoff=rad_cutoff_2D, verbose=False)
        temp_adata.uns['Spatial_Net']['SNN'] = temp_section
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' %
                  (temp_adata.uns['Spatial_Net'].shape[0]/temp_adata.n_obs))
        adata.uns['Spatial_Net_2D'] = pd.concat(
            [adata.uns['Spatial_Net_2D'], temp_adata.uns['Spatial_Net']])
    for it in range(num_section-1):
        section_1 = section_order[it]
        section_2 = section_order[it+1]
        if verbose:
            print('------Calculating SNN between adjacent section %s and %s.' %
                  (section_1, section_2))
        Z_Net_ID = section_1+'-'+section_2
        temp_adata = adata[adata.obs[key_section].isin(
            [section_1, section_2]), ]
        Cal_Spatial_Net(
            temp_adata, rad_cutoff=rad_cutoff_Zaxis, verbose=False)
        spot_section_trans = dict(
            zip(temp_adata.obs.index, temp_adata.obs[key_section]))
        temp_adata.uns['Spatial_Net']['Section_id_1'] = temp_adata.uns['Spatial_Net']['Cell1'].map(
            spot_section_trans)
        temp_adata.uns['Spatial_Net']['Section_id_2'] = temp_adata.uns['Spatial_Net']['Cell2'].map(
            spot_section_trans)
        used_edge = temp_adata.uns['Spatial_Net'].apply(
            lambda x: x['Section_id_1'] != x['Section_id_2'], axis=1)
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[used_edge, ]
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[:, [
            'Cell1', 'Cell2', 'Distance']]
        temp_adata.uns['Spatial_Net']['SNN'] = Z_Net_ID
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' %
                  (temp_adata.uns['Spatial_Net'].shape[0]/temp_adata.n_obs))
        adata.uns['Spatial_Net_Zaxis'] = pd.concat(
            [adata.uns['Spatial_Net_Zaxis'], temp_adata.uns['Spatial_Net']])
    adata.uns['Spatial_Net'] = pd.concat(
        [adata.uns['Spatial_Net_2D'], adata.uns['Spatial_Net_Zaxis']])
    if verbose:
        print('3D SNN contains %d edges, %d cells.' %
            (adata.uns['Spatial_Net'].shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %
            (adata.uns['Spatial_Net'].shape[0]/adata.n_obs))

def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STMGraph', random_seed=52, dist=None):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    refined_pred=refine(adata)
    adata.obs["refine_mclust"]=refined_pred
    adata.obs["refine_mclust"]=adata.obs["refine_mclust"].astype('category')
    return adata

def refine(adata=None):
    refined_pred=[]
    dis_df=adata.uns['Spatial_Net'].reset_index(drop=True)
    sample_id=adata.obs.index.tolist()
    pred=adata.obs['mclust'].tolist()
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    for index in sample_id:
        num_index=dis_df[dis_df.loc[:,'Cell1']==index].index
        num_nbs=len(num_index)
        self_pred=pred.loc[index, "pred"]
        if num_nbs>0:
            dis_tmp=dis_df.loc[num_index,:]
            nbs_pred=pred.loc[dis_tmp.loc[:,'Cell2'].to_list(), "pred"]
           
            v_c=nbs_pred.value_counts()
            if self_pred in v_c.index:
                if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
                    refined_pred.append(v_c.idxmax())
                else:           
                    refined_pred.append(self_pred)
            else:
                refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred

