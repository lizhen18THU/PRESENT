import argparse
import pandas as pd
import scanpy as sc
import os
from PRESENT import gene_sets_alignment, peak_sets_alignment
from PRESENT import PRESENT_BC_function, run_leiden

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PRESENT-BC: multi-sample integration of spatially resolved omics data')
    parser.add_argument('--outputdir', type=str, default='./PRESENT_output', help='The output dir for PRESENT')
    parser.add_argument('--spatial_key', type=str, default='spatial', help='adata_rna/adata_atac/adata_adt.obsm key under which to load the spatial matrix of spots')
    parser.add_argument('--batch_key', type=str, default='batch', help='adata_rna/adata_atac/adata_adt.obs key under which to load the batch indices of spots')
    parser.add_argument('--adata_rna_path_list', type=str, nargs='*', default=[], help='The path to RNA raw count matrices of spots in anndata.AnnData format')
    parser.add_argument('--adata_atac_path_list', type=str, nargs='*', default=[], help='The path to ATAC raw fragment count matrices of spots in anndata.AnnData format')
    parser.add_argument('--adata_adt_path_list', type=str, nargs='*', default=[], help='The path to ADT raw count matrices of spots in anndata.AnnData format')

    parser.add_argument('--nclusters', type=int, default=10, help='Number of spatial clusters')
    parser.add_argument('--gene_min_cells', type=int, default=1, help='Minimum number of cells expressed required for a gene to pass filtering')
    parser.add_argument('--peak_min_cells_fraction', type=float, default=0.03, help='Minimum fraction of cells accessible required for a peak to pass filtering')
    parser.add_argument('--protein_min_cells', type=int, default=1, help='Minimum number of cells expressed required for a protein to pass filtering')
    parser.add_argument('--num_hvg', type=int, default=3000, help='Number of highly variable genes to select for RNA data')
    parser.add_argument('--d_lat', type=int, default=50, help='The latent dimension of final embeddings')
    parser.add_argument('--intra_neighbors', type=int, default=6, help='Number of intra_neighbors for each spot to construct cross-sample graph')
    parser.add_argument('--inter_neighbors', type=int, default=6, help='Number of inter_neighbors for each spot to construct cross-sample graph')
    parser.add_argument('--epochs', type=float, default=100, help='Max epochs to train the model')
    parser.add_argument('--lr', type=int, default=0.001, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=320, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda', help='Device used for training')
    parser.add_argument('--device_id', type=int, default=0, help='Which gpu is used for training')

    args = parser.parse_args()
    adata_rna = adata_atac = adata_adt = None

    print("Unifying genomic feaures for different omics layers...")
    if len(args.adata_rna_path_list)==1: 
        adata_rna = sc.read_h5ad(args.adata_rna_path_list[0])
    elif len(args.adata_rna_path_list)>1: 
        adata_rna_list = []
        for path in args.adata_rna_path_list:
            adata_rna_list.append(sc.read_h5ad(path))
        adata_rna_list = gene_sets_alignment(adata_rna_list)
        adata_rna = adata_rna_list[0].concatenate(adata_rna_list[1:])
        adata_rna.obs[args.batch_key] = adata_rna.obs["batch"]

    if len(args.adata_atac_path_list)==1: 
        adata_atac = sc.read_h5ad(args.adata_atac_path_list[0])
    elif len(args.adata_atac_path_list)>1: 
        adata_atac_list = []
        for path in args.adata_atac_path_list:
            adata_atac_list.append(sc.read_h5ad(path))
        adata_atac_list = peak_sets_alignment(adata_atac_list)
        adata_atac = adata_atac_list[0].concatenate(adata_atac_list[1:])
        adata_atac.obs[args.batch_key] = adata_atac.obs["batch"]
    
    if len(args.adata_adt_path_list)==1: 
        adata_adt = sc.read_h5ad(args.adata_adt_path_list[0])
    elif len(args.adata_adt_path_list)>1: 
        adata_adt_list = []
        for path in args.adata_adt_path_list:
            adata_adt_list.append(sc.read_h5ad(path))
        adata_adt_list = gene_sets_alignment(adata_adt_list)
        adata_adt = adata_adt_list[0].concatenate(adata_adt_list[1:])
        adata_adt.obs[args.batch_key] = adata_adt.obs["batch"]
    print("Genomic features have been unified")

    adata = PRESENT_BC_function(
        args.spatial_key, 
        args.batch_key,
        adata_rna = adata_rna,
        adata_atac = adata_atac,
        adata_adt = adata_adt,

        gene_min_cells = args.gene_min_cells,
        peak_min_cells_fraction = args.peak_min_cells_fraction,
        protein_min_cells = args.protein_min_cells,
        num_hvg = args.num_hvg,
        nclusters = args.nclusters,
        d_lat = args.d_lat,
        intra_neighbors = args.intra_neighbors,
        inter_neighbors = args.inter_neighbors,
        epochs = args.epochs,
        lr = args.lr,
        batch_size = args.batch_size,
        device = args.device,
        device_id = args.device_id
    )
    if args.outputdir.endswith("/"): args.outputdir = args.outputdir[0:-1]
    os.makedirs(args.outputdir, exist_ok=True)

    adata.write_h5ad(args.outputdir + "/adata_output.h5ad")
    pd.DataFrame(adata.X, index=adata.obs_names).to_csv(args.outputdir + "/embeddings_output.csv")
    adata.obs.loc[:, ["LeidenClusters"]].to_csv(args.outputdir + "/domains_output.csv")
    print("Joint embeddings and identified domains saved")
