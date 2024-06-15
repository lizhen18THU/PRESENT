# Cross-modality representation and multi-sample integration of spatially resolved omics data

## Overview

Spatially resolved sequencing technologies have revolutionized the characterization of biological regulatory processes within microenvironment by simultaneously accessing the states of genomic regions, genes and proteins, along with the spatial coordinates of cells, necessitating advanced computational methods for the cross-modality and multi-sample integrated analysis of spatial omics datasets. To address this gap, we propose PRESENT, an effective and scalable contrastive learning framework, for the cross-modality representation of spatially resolved omics data. Through comprehensive experiments on massive spatially resolved datasets, PRESENT achieves superior performance across various species, tissues, and sequencing technologies, including spatial epigenomics, transcriptomics, and multi-omics. Specifically, PRESENT empowers the incorporation of spatial dependency and complementary omics information simultaneously, facilitating the detection of spatial domains and uncovering biological regulatory mechanisms within microenvironment. Furthermore, PRESENT can be extended to the integrative analysis of horizontal and vertical samples across different dissected regions or developmental stages, thereby promoting the identification of hierarchical structures from a spatiotemporal perspective.

<div align=center>
<img src = "docs/source/PRESENT_Overview.png" width = 100% height = 100%>
</div>


## Installation

### Dependencies
```
numpy==1.24.4
pandas==2.0.3
scipy==1.9.3
scikit-learn==1.3.2
louvain==0.8.0,
leidenalg==0.9.1,
anndata==0.9.2
networkx==3.1
scanpy==1.9.8
episcanpy==0.3.2
genomicranges==0.4.2
iranges==0.2.1
biocutils==0.1.3
torch==2.0.0
torch-geometric==2.3.1
```

### Installation via pypi
PRESENT is available on PyPI [here](https://pypi.org/project/bio-present) and can be installed via
```
pip install bio-present
pip install pyg_lib==0.2.0 torch_sparse==0.6.17 torch_cluster==1.6.1 torch_scatter==2.1.1 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Installation via Github
You can also install PRESENT from GitHub via
```
git clone https://github.com/lizhen18THU/PRESENT.git
cd PRESENT
python setup.py install
pip install pyg_lib==0.2.0 torch_sparse==0.6.17 torch_cluster==1.6.1 torch_scatter==2.1.1 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Quick start

The input of different spatial omics layers should be raw count matrices in [anndata.AnnData](https://anndata.readthedocs.io/en/latest/) format, where spatial coordinate matrix is stored in the `adata.obsm[spatial_key]` of corresponding AnnData object. The output including the joint representations and the identified domains, stored in an AnnData file and two csv files.

### Cross-modality representation of a single sample with PRESENT

#### Run PRESENT on spatial transcriptomics data

Suppose we have a spatial transcriptomics data, where the RNA raw count matrix and the spatial coordinate matrix are stored in file data/adata_rna.h5ad. The command to run PRESENT model on the spatial multi-omics data is
```
python3 PRESENT_script.py --outputdir ./PRESENT_output --spatial_key spatial --adata_rna_path data/adata_rna.h5ad --gene_min_cells 1  --num_hvg 3000 --nclusters 10
```

Optionally, PRESENT also can integrate reference data to address datasets with low sequencing depth and signal-to-noise ratio. Suppose the reference transcriptomic data are stored in data/rdata_rna.h5ad and the annotation is stored in the rdata_rna.obs["domains"], then the command to run PRESENT model is
```
python3 PRESENT_script.py --outputdir ./PRESENT_output --spatial_key spatial --adata_rna_path data/adata_rna.h5ad --rdata_rna_path data/rdata_rna.h5ad --rdata_rna_anno domains --gene_min_cells 1  --num_hvg 3000 --nclusters 10
```

#### Run PRESENT on spatial ATAC-seq data
Similarly, the command to run PRESENT model on the spatial ATAC-seq data is
```
python3 PRESENT_script.py --outputdir ./PRESENT_output --spatial_key spatial --adata_atac_path data/adata_atac.h5ad --peak_min_cells_fraction 0.03 --nclusters 10
```

Optionally, 
```
python3 PRESENT_script.py --outputdir ./PRESENT_output --spatial_key spatial --adata_atac_path data/adata_atac.h5ad --rdata_rna_path data/rdata_rna.h5ad --rdata_rna_anno domains --peak_min_cells_fraction 0.03 --nclusters 10
```

#### Run PRESENT on spatial multi-omics co-profiling data
Suppose we have a spatial RNA-ATAC co-profiling data, where the RNA and ATAC raw count matrix are stored in paired files data/adata_rna.h5ad and data/adata_atac.h5ad.
```
python3 PRESENT_script.py --outputdir ./PRESENT_output --spatial_key spatial --adata_rna_path data/adata_rna.h5ad --gene_min_cells 1  --num_hvg 3000 --adata_atac_path data/adata_atac.h5ad --peak_min_cells_fraction 0.03 --nclusters 10 
```

Similarly, for spatial RNA-ADT co-profiling data, the command denotes
```
python3 PRESENT_script.py --outputdir ./PRESENT_output --spatial_key spatial --adata_rna_path data/adata_rna.h5ad --gene_min_cells 1  --num_hvg 3000 --adata_adt_path data/adata_adt.h5ad --protein_min_cells 1 --nclusters 10 
```

### Multi-sample integration with PRESENT-BC

#### Run PRESENT-BC for the integration of spatial transcriptomics data

Suppose we have two spatial transcriptomics samples for integrative analysis, where the two RNA raw count matrices  are stored in separated AnnData files data/adata_rna1.h5ad and data/adata_rna2.h5ad. The command to run PRESENT-BC model is
```
python3 PRESENT_BC_script.py --outputdir ./PRESENT_output --spatial_key spatial --adata_rna_path_list data/adata_rna1.h5ad data/adata_rna2.h5ad --gene_min_cells 1  --num_hvg 3000 --nclusters 10
```
If the feature sets of different samples have already been unified and the expression matrices have been concatnated into a single AnnData file data/adata_rna_concatnated.h5ad, and the batch indices are stored in 'adata_rna_concatnated.obs["batch"]', then
```
python3 PRESENT_BC_script.py --outputdir ./PRESENT_output --spatial_key spatial --batch_key batch --adata_rna_path_list data/adata_rna_concatnated.h5ad --gene_min_cells 1  --num_hvg 3000 --nclusters 10
```

#### Run PRESENT-BC for the integration of spatial ATAC-seq data
Similarly, the command to run PRESENT-BC model for the integration of spatial ATAC-seq data is
```
python3 PRESENT_BC_script.py --outputdir ./PRESENT_output --spatial_key spatial --adata_atac_path_list data/adata_atac1.h5ad data/adata_atac2.h5ad --peak_min_cells_fraction 0.03 --nclusters 10
```
and
```
python3 PRESENT_BC_script.py --outputdir ./PRESENT_output --spatial_key spatial --batch_key batch --adata_atac_path_list data/adata_atac_concatnated.h5ad --peak_min_cells_fraction 0.03 --nclusters 10
```

#### Run PRESENT-BC for the integration of spatial multi-omics co-profiling data
Suppose we have a spatial RNA-ATAC co-profiling data, where the RNA and ATAC raw count matrix of different samples are stored in files data/adata_rna1.h5ad, data/adata_rna2.h5ad, data/adata_atac1.h5ad and data/adata_atac2.h5ad. Then
```
python3 PRESENT_BC_script.py --outputdir ./PRESENT_output --spatial_key spatial --adata_rna_path_list data/adata_rna1.h5ad data/adata_rna2.h5ad --gene_min_cells 1  --num_hvg 3000 --adata_atac_path_list data/adata_atac1.h5ad data/adata_atac2.h5ad --peak_min_cells_fraction 0.03 --nclusters 10 
```
If the feature sets of different samples have already been unified and the expression matrices of each omics layer have been concatnated into a single AnnData file, then
```
python3 PRESENT_BC_script.py --outputdir ./PRESENT_output --spatial_key spatial --batch_key batch --adata_rna_path_list data/adata_rna_concatnated.h5ad --gene_min_cells 1  --num_hvg 3000 --adata_atac_path_list data/adata_atac_concatnated.h5ad --peak_min_cells_fraction 0.03 --nclusters 10 
```

### Explanations for the arguments of scripts PRESENT_script.py and PRESENT-BC.py
+ {--outputdir}: A path specifying where the final results are stored, default: ./PRESENT_output
+ {--spatial_key}: adata.obsm key under which to load the spatial matrix in the AnnData object, default: spatial
+ {--batch_key}: adata.obs key under which to load the batch indices in the AnnData object, default: batch
+ {--nclusters}: Number of spatial clusters, default: 10
+ {--gene_min_cells}: Minimum number of cells expressed required for a gene to pass filtering, default: 1
+ {--peak_min_cells_fraction}: Minimum fraction of cells accessible required for a peak to pass filtering, default: 0.03
+ {--protein_min_cells}: Minimum number of cells expressed required for a protein to pass filtering, default: 1
+ {--num_hvg}: Number of highly variable genes to select for RNA data, default: 3000
+ {--d_lat}: The latent dimension of final embeddings, default: 50
+ {--k_neighbors}: Number of neighbors for each spot to construct graph, default: 6
+ {--epochs}: Max epochs to train the model, default: 100
+ {--lr}: Initial learning rate, default: 0.001
+ {--batch_size}: Batch size for training, default: 320
+ {--device}: Device used for training (cuda or cpu), default: cuda
+ {--device_id}: Which gpu to use for training

## Citation

Zhen Li, Xuejian Cui, Xiaoyang Chen, Zijing Gao, Yuyao Liu, Yan Pan, Shengquan Chen and Rui Jiang. "Cross-modality representation and multi-sample integration of spatially resolved omics data." Preprint at bioRxiv https://doi.org/10.1101/2024.06.10.598155 (2024).
