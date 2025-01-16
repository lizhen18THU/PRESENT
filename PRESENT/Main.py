import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import torch
import episcanpy.api as epi

from .Model import *
from .Utils import run_leiden

def PRESENT_function(
        spatial_key: str="spatial",
        batch_key: Union[str, NoneType] = None,
        adata_rna: Union[str, ad.AnnData, NoneType] = None,
        adata_atac: Union[str, ad.AnnData, NoneType] = None, 
        adata_adt: Union[str, ad.AnnData, NoneType] = None,
        rdata_rna: Union[str, ad.AnnData, NoneType] = None,
        rdata_rna_anno: Union[str, NoneType] = None,
        rdata_atac: Union[str, ad.AnnData, NoneType] = None, 
        rdata_atac_anno: Union[str, NoneType] = None,
        rdata_adt: Union[str, ad.AnnData, NoneType] = None,
        rdata_adt_anno: Union[str, NoneType] = None,
        gene_min_cells: int = 1,
        peak_min_cells_fraction: float = 0.03,
        protein_min_cells: int = 1,
        num_hvg: int = 3000,
        nclusters: int = 10,
        d_lat: int = 50,
        k_neighbors: int = 6,
        intra_neighbors: int = 6,
        inter_neighbors: int = 6,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int=320,
        device: str="cuda",
        device_id: int=0,
    ) -> ad.AnnData:
    """
    PRESENT: Cross-modality representation and multi-sample integration of spatially resolved omics data

    Parameters
    ------
    spatial_key
        adata_rna/adata_atac/adata_adt.obsm key under which to load the spatial matrix of spots
    batch_key
        adata_rna/adata_atac/adata_adt.obs key under which to load the batch indices of spots
    adata_rna
        The RNA raw count matrix of spots in anndata.AnnData format
    adata_atac
        The ATAC raw fragment count matrix of spots in anndata.AnnData format
    adata_adt
        The ADT raw count matrix of spots in anndata.AnnData format
    rdata_rna
        The RNA raw counts of reference data in anndata.AnnData format
    rdata_rna_anno
        rdata_rna.obs key under which to load the annotation
    rdata_atac
        The ATAC raw fragment counts of reference data in anndata.AnnData format
    rdata_atac_anno
        rdata_atac.obs key under which to load the annotation
    rdata_adt
        The ADT raw counts of reference data in anndata.AnnData format
    rdata_adt_anno
        rdata_adt.obs key under which to load the annotation
    gene_min_cells
        Minimum number of cells expressed required for a gene to pass filtering
    peak_min_cells_fraction
        Minimum fraction of cells accessible required for a peak to pass filtering
    protein_min_cells
        Minimum number of cells expressed required for a protein to pass filtering
    num_hvg
        Number of highly variable genes to select for RNA data
    nclusters
        Number of spatial clusters
    d_lat
        The latent dimension of final embeddings
    k_neighbors
        Number of neighbors for each spot to construct graph
    intra_neighbors
        Number of intra_neighbors for each spot to construct cross-sample graph
    inter_neighbors
        Number of inter_neighbors for each spot to construct cross-sample graph
    epochs
        Max epochs to train the model
    lr
        Initial learning rate
    batch_size
        Batch size for training
    device
        Device used for training
    device_id
        Which gpu is used for training

    Return
    ------
    adata
        AnnData containing the joint latent embeddings in adata.obsm['embeddings'] and the identified spatial domains in adata.obs['LeidenClusters']
    """

    print("Loading data and parameters...")
    device = torch.device("cuda:" + str(device_id)) if device.lower() == "cuda" else torch.device("cpu")
    assert gene_min_cells >= 0, "gene_min_cells should not be less than 0"
    assert protein_min_cells >= 0, "protein_min_cells should not be less than 0"
    assert peak_min_cells_fraction >= 0 and peak_min_cells_fraction <= 1, "protein_min_cells should not be less than 0 or more than 1"
    assert k_neighbors > 0, "k_neighbors should be larger than 0"
    assert intra_neighbors > 0, "intra_neighbors should be larger than 0"
    assert inter_neighbors > 0, "inter_neighbors should be larger than 0"

    if adata_rna is not None and isinstance(adata_rna, str):
        adata_rna = sc.read_h5ad(adata_rna)
    if adata_atac is not None and isinstance(adata_atac, str):
        adata_atac = sc.read_h5ad(adata_atac)
    if adata_adt is not None and isinstance(adata_adt, str):
        adata_adt = sc.read_h5ad(adata_adt)

    if rdata_rna is not None and isinstance(rdata_rna, str):
        rdata_rna = sc.read_h5ad(rdata_rna)
    if rdata_atac is not None and isinstance(rdata_atac, str):
        rdata_atac = sc.read_h5ad(rdata_atac)
    if rdata_adt is not None and isinstance(rdata_adt, str):
        rdata_adt = sc.read_h5ad(rdata_adt)

    index = None
    if adata_rna is not None:
        sc.pp.filter_genes(adata_rna, min_cells=gene_min_cells)
        sc.pp.filter_cells(adata_rna, min_genes=1)
        sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=num_hvg, subset=True)

        if rdata_rna is not None: 
            rdata_rna, adata_rna = ref_feature_alignment(rdata_rna, adata_rna, omics="RNA")
            if rdata_rna_anno is not None: rdata_rna_anno = rdata_rna.obs[rdata_rna_anno].values

        sc.pp.filter_genes(adata_rna, min_cells=1)
        sc.pp.filter_cells(adata_rna, min_genes=1)
        index = adata_rna.obs_names

    if adata_atac is not None:
        epi.pp.filter_features(adata_atac, min_cells=int(adata_atac.shape[0] * peak_min_cells_fraction))
        epi.pp.filter_cells(adata_atac, min_features=1)

        if rdata_atac is not None: 
            rdata_atac, adata_atac = ref_feature_alignment(rdata_atac, adata_atac, omics="ATAC")
            if rdata_atac_anno is not None: rdata_atac_anno = rdata_atac.obs[rdata_atac_anno].values


        epi.pp.filter_features(adata_atac, min_cells=1)
        epi.pp.filter_cells(adata_atac, min_features=1)
        index = np.intersect1d(adata_atac.obs_names, index) if index is not None else adata_atac.obs_names

    if adata_adt is not None:
        sc.pp.filter_genes(adata_adt, min_cells=protein_min_cells)
        sc.pp.filter_cells(adata_adt, min_genes=1)

        if rdata_adt is not None: 
            rdata_adt, adata_adt = ref_feature_alignment(rdata_adt, adata_adt, omics="ADT")
            if rdata_adt_anno is not None: rdata_adt_anno = rdata_adt.obs[rdata_adt_anno].values

        sc.pp.filter_genes(adata_adt, min_cells=1)
        sc.pp.filter_cells(adata_adt, min_genes=1)
        index = np.intersect1d(adata_adt.obs_names, index) if index is not None else adata_adt.obs_names
    assert index is not None, "Please input at least one omics layer in anndata.AnnData format"

    spatial_mtx = batch_indices = None
    rna_dim = cas_dim = adt_dim = None
    if adata_rna is not None: 
        adata_rna = adata_rna[index, :]
        rna_dim = adata_rna.shape[1]
        if batch_key is not None and batch_key in adata_rna.obs.keys(): batch_indices = adata_rna.obs[batch_key].values.astype(str)
        if spatial_key in adata_rna.obsm.keys(): spatial_mtx = adata_rna.obsm[spatial_key]
    if adata_atac is not None: 
        adata_atac = adata_atac[index, :]
        cas_dim = adata_atac.shape[1]
        if batch_key is not None and batch_key in adata_atac.obs.keys(): batch_indices = adata_atac.obs[batch_key].values.astype(str)
        if spatial_key in adata_atac.obsm.keys(): spatial_mtx = adata_atac.obsm[spatial_key]
    if adata_adt is not None: 
        adata_adt = adata_adt[index, :]
        adt_dim = adata_adt.shape[1]
        if batch_key is not None and batch_key in adata_adt.obs.keys(): batch_indices = adata_adt.obs[batch_key].values.astype(str)
        if spatial_key in adata_adt.obsm.keys(): spatial_mtx = adata_adt.obsm[spatial_key]
    assert spatial_mtx is not None, "Please provide spatial mtx in the adata_rna/adata_atac/adata_adt.obsm under spatial_key"
    if batch_key is not None: assert batch_indices is not None, "Invalid batch_key: cannot find batch_key in adata.obs"
    if batch_key is not None and batch_indices is not None: n_batches = np.unique(batch_indices).shape[0]

    rna_counts=adata_rna.X if adata_rna is not None else None
    cas_counts=adata_atac.X if adata_atac is not None else None
    adt_counts=adata_adt.X if adata_adt is not None else None

    ref_rna_counts = rdata_rna.X if rdata_rna is not None else None
    ref_cas_counts = rdata_atac.X if rdata_atac is not None else None
    ref_adt_counts = rdata_adt.X if rdata_adt is not None else None

    print("Input data has been loaded")

    if batch_key is not None and batch_indices is not None:
        model = PRESENT_BC(rna_dim=rna_dim, cas_dim=cas_dim, adt_dim=adt_dim, n_batches=n_batches,
                           d_lat=d_lat, intra_neighbors=intra_neighbors, inter_neighbors=inter_neighbors).to(device)
        embeddings, omics_lat, omics_impute = model.model_train(spa_mat=spatial_mtx, 
                                                                rna_counts=rna_counts, ref_rna_counts=ref_rna_counts, ref_rna_anno=rdata_rna_anno,
                                                                cas_counts=cas_counts, ref_cas_counts=ref_cas_counts, ref_cas_anno=rdata_atac_anno,
                                                                adt_counts=adt_counts, ref_adt_counts=ref_adt_counts, ref_adt_anno=rdata_adt_anno,
                                                                batch_label=batch_indices,
                                                                impute=False,
                                                                epochs=epochs, lr=lr, batch_size=batch_size, device=device)
    else: 
        model = PRESENT_RP(rna_dim=rna_dim, cas_dim=cas_dim, adt_dim=adt_dim,
                        d_lat=d_lat, k_neighbors=k_neighbors).to(device)
        embeddings, omics_lat, omics_impute = model.model_train(spa_mat=spatial_mtx, 
                                                                rna_counts=rna_counts, ref_rna_counts=ref_rna_counts, ref_rna_anno=rdata_rna_anno,
                                                                cas_counts=cas_counts, ref_cas_counts=ref_cas_counts, ref_cas_anno=rdata_atac_anno,
                                                                adt_counts=adt_counts, ref_adt_counts=ref_adt_counts, ref_adt_anno=rdata_adt_anno,
                                                                impute=False,
                                                                epochs=epochs, lr=lr, batch_size=batch_size, device=device)
    if adata_rna is not None:
        adata = sc.AnnData(pd.DataFrame(embeddings, index=index), obs=adata_rna.obs.copy(), obsm=adata_rna.obsm)
    elif adata_atac is not None:
        adata = sc.AnnData(pd.DataFrame(embeddings, index=index), obs=adata_atac.obs.copy(), obsm=adata_atac.obsm)
    else:
        adata = sc.AnnData(pd.DataFrame(embeddings, index=index), obs=adata_adt.obs.copy(), obsm=adata_adt.obsm)
    adata.obsm["embeddings"] = embeddings
    adata = run_leiden(adata, n_cluster=nclusters, use_rep="embeddings", key_added="LeidenClusters")
    return adata
