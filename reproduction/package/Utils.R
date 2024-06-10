library(Seurat)
library(Matrix)
library(anndata)

LoadSparseMatrix <- function(txt_file, dims){
    exp_mtx = read.table(txt_file, header=T, sep=",", row.names=1)
    exp_mtx = sparseMatrix(i=exp_mtx$i+1, j=exp_mtx$j+1, x=exp_mtx$x, dims=dims)
    
    return(exp_mtx)
}

LoadMetaData <- function(csv_file){
    meta = read.table(csv_file, header=T, sep=",", row.names=1) 
    return(meta)
}

LoadSeuratFromDIR <- function(data_folder){
    cell_meta = read.table(paste0(data_folder, "/cell_meta.csv"), header=T, sep=",", row.names=1)
    gene_meta = read.table(paste0(data_folder, "/gene_meta.csv"), header=T, sep=",", row.names=1)
    exp_mtx = read.table(paste0(data_folder, "/counts.txt"), header=T, sep=",", row.names=1)
    exp_mtx = sparseMatrix(i=exp_mtx$i+1, j=exp_mtx$j+1, x=exp_mtx$x, dims=c(nrow(cell_meta), nrow(gene_meta)))
    rownames(exp_mtx) = rownames(cell_meta)
    colnames(exp_mtx) = rownames(gene_meta)

    seu.obj = CreateSeuratObject(counts=t(exp_mtx), project="seu", assay="counts")
    seu.obj = AddMetaData(object = seu.obj, metadata = cell_meta)
    
    return(seu.obj)
}

h5seurat2seu <- function(h5seu.file){
    library(SeuratData)
    library(SeuratDisk)
    library(rhdf5)
    sobj <- LoadH5Seurat(file=h5seu.file,  meta.data = FALSE, misc = FALSE)
    obs <- h5read(h5seu.file, "/meta.data")

    meta <- data.frame(lapply(names(obs), function(x) { 
      if (length(obs[[x]])==2) 
        obs[[x]][['categories']][ifelse(obs[[x]][['codes']] >= 0, obs[[x]][['codes']] + 1, NA)]
      else 
        as.numeric(obs[[x]])
    }
    ), row.names=Cells(sobj))
    colnames(meta) <- names(obs)

    sobj <- AddMetaData(sobj,meta)
    return(sobj)
}

label_mapping <- function(vec, map_dic){
    return(as.vector(factor(vec, levels=names(map_dic), labels = map_dic)))
}

LoadSeuratFromAnndata <- function(anndata_file, spatial=T){
    adata = read_h5ad(anndata_file)
    counts = as(t(adata$X), "CsparseMatrix")
    colnames(counts) = rownames(adata$obs)
    rownames(counts) = rownames(adata$var)
    
    seu.obj = CreateSeuratObject(counts=counts, project="seu", assay="counts")
    cell_meta = adata$obs
    if (ncol(cell_meta)>0){
        seu.obj = AddMetaData(object = seu.obj, metadata = cell_meta)
    }
    
    return(seu.obj)
}

default_louvain <- function(seu, graph.name=NULL, key_added="Dlouvain"){
    if(!is.null(graph.name)){
        seu <- FindClusters(seu, algorithm=1, graph.name=graph.name, resolution=1.0)
    }else{
        seu <- FindClusters(seu, algorithm=1, resolution=1.0)
    }
    seu[[key_added]] = seu[["seurat_clusters"]]
    
    return(seu)
}

default_leiden <- function(seu, graph.name=NULL, key_added="Dleiden"){
    if(!is.null(graph.name)){
        seu <- FindClusters(seu, algorithm=4, graph.name=graph.name, resolution=1.0)
    }else{
        seu <- FindClusters(seu, algorithm=4, resolution=1.0)
    }
    seu[[key_added]] = seu[["seurat_clusters"]]
    
    return(seu)
}

run_louvain <- function(seu, n_cluster, graph.name=NULL, range_min=0, range_max=3, max_steps=30, tolerance=0, key_added="Nlouvain"){
    this_step = 0
    this_min = range_min
    this_max = range_max
    while(this_step < max_steps){
        this_resolution = this_min + ((this_max-this_min)/2)
        if(!is.null(graph.name)){
            seu <- FindClusters(seu, resolution = this_resolution, graph.name=graph.name, algorithm=1)
        }else{
            seu <- FindClusters(seu, resolution = this_resolution, algorithm=1)
        }
        this_clusters = nrow(unique(seu[["seurat_clusters"]]))
        
        if(this_clusters>n_cluster){
            this_max = this_resolution
        }else if(this_clusters<n_cluster){
            this_min = this_resolution
        }else{
            seu[[key_added]] = seu[["seurat_clusters"]]
            return(seu)
        }
        this_step=this_step+1
    }
    print("Cannot find the number of clusters")
    seu[[key_added]] = seu[["seurat_clusters"]]
    return(seu)
}

run_leiden <- function(seu, n_cluster, graph.name=NULL, range_min=0, range_max=3, max_steps=30, tolerance=0, key_added="Nleiden"){
    this_step = 0
    this_min = range_min
    this_max = range_max
    while(this_step<max_steps){
        this_resolution = this_min + ((this_max-this_min)/2)
        if(!is.null(graph.name)){
            seu <- FindClusters(seu, resolution = this_resolution, graph.name=graph.name, algorithm=4)
        }else{
            seu <- FindClusters(seu, resolution = this_resolution, algorithm=4)
        }
        this_clusters = nrow(unique(seu[["seurat_clusters"]]))
        
        if(this_clusters>n_cluster+tolerance){
            this_max = this_resolution
        }else if(this_clusters<n_cluster-tolerance){
            this_min = this_resolution
        }else{
            seu[[key_added]] = seu[["seurat_clusters"]]
            return(seu)
        }
        this_step=this_step+1
    }
    print("Cannot find the number of clusters")
    seu[[key_added]] = seu[["seurat_clusters"]]
    return(seu)
}

mclust_R <- function(seu, n_cluster, model_name="EEE", random_seed=666, reduction="embeddings", key_added="Nmclust"){
    set.seed(random_seed)
    mtx = Embeddings(object = seu, reduction = reduction)
    result = mclust::Mclust(as.matrix(mtx), num_cluster=n_cluster, modelNames=model_name)
    labels = as.vector(result$classification)
    seu[[key_added]] = label
    
    return(seu)
}

cluster_metrics <- function(truth,pred){
    sklearn_metrics <- reticulate::import("sklearn.metrics")
    fowlkes_mallows_score <- sklearn_metrics$cluster$fowlkes_mallows_score
    adjusted_rand_score  <- sklearn_metrics$cluster$adjusted_rand_score
    normalized_mutual_info_score  <- sklearn_metrics$cluster$normalized_mutual_info_score
    homogeneity_score  <- sklearn_metrics$cluster$homogeneity_score
    adjusted_mutual_info_score  <- sklearn_metrics$cluster$adjusted_mutual_info_score
    completeness_score  <- sklearn_metrics$cluster$completeness_score
    
    if(typeof(pred)=="list"){
        pred=unlist(pred[1])
    }
    if(typeof(truth)=="list"){
        truth=as.vector(unlist(truth[1]))
    }
    pred = as.vector(pred)
    truth = as.vector(truth)
    
    ari=adjusted_rand_score(truth,pred)
    ami=adjusted_mutual_info_score(truth,pred)
    nmi=normalized_mutual_info_score(truth,pred)
    fmi=fowlkes_mallows_score(truth,pred)
    comp=completeness_score(truth,pred)
    homo=homogeneity_score(truth,pred)
    print(paste0("ARI:",round(ari,3),",  AMI: ",round(ami,3),",  NMI: ",round(nmi,3),",  FMI: ",round(fmi,3),",  Comp: ",round(comp,3),",  Homo: ",round(homo,3)))
    
    return(c(ari, ami, nmi, fmi, comp, homo))
}