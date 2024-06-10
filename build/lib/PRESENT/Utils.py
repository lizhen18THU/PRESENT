import scipy
import os
import random
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean
from anndata import AnnData
from typing import Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

def match_cluster_labels(true_labels, est_labels):
    def _match_cluster_labels(true_labels, est_labels):
        import networkx as nx
        true_labels_arr = np.array(list(true_labels))
        est_labels_arr = np.array(list(est_labels))

        org_cat = list(np.sort(list(pd.unique(true_labels))))
        est_cat = list(np.sort(list(pd.unique(est_labels))))

        B = nx.Graph()
        B.add_nodes_from([i + 1 for i in range(len(org_cat))], bipartite=0)
        B.add_nodes_from([-j - 1 for j in range(len(est_cat))], bipartite=1)

        for i in range(len(org_cat)):
            for j in range(len(est_cat)):
                weight = np.sum((true_labels_arr == org_cat[i]) * (est_labels_arr == est_cat[j]))
                B.add_edge(i + 1, -j - 1, weight=-weight)

        match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)

        if len(org_cat) >= len(est_cat):
            return np.array([match[-est_cat.index(c) - 1] - 1 for c in est_labels_arr])
        else:
            unmatched = [c for c in est_cat if not (-est_cat.index(c) - 1) in match.keys()]
            l = []
            for c in est_labels_arr:
                if (-est_cat.index(c) - 1) in match:
                    l.append(match[-est_cat.index(c) - 1] - 1)
                else:
                    l.append(len(org_cat) + unmatched.index(c))
            return np.array(l)
    
    new_idx = _match_cluster_labels(true_labels, est_labels)
    if len(np.unique(true_labels)) < len(np.unique(new_idx)):
        index = np.array(np.unique(true_labels).tolist() + [str(i) for i in range(len(np.unique(new_idx)) - len(np.unique(true_labels)))])
    else:
        index = np.unique(true_labels)
    
    return index[new_idx]
    
def construct_pseudo_bulk(adata_ref, key, min_samples=11):
    """
    construct pseudo bulk from reference dataset according to annotation

    Parameters
    ------
    adata_ref
        reference dataset of anndata format
    key
        key of the annoatation
    min_samples
        minimum number of pseudo bulk samples should be constructed from reference dataset

    Returns
    ------
    scanpy.anndata
        pseudo bulk samples of anndata format
    """
    print("Construct pseudo bulk reference data...")
    index_list = []
    key_index = adata_ref.obs[key].values
    bulks = np.unique(key_index)
    df_index = []
    if min_samples is not None: min_samples += 1
    if min_samples is not None and min_samples > bulks.shape[0]:
        times = min_samples // bulks.shape[0]
        r = (1 / times + 1) / 2
        print("DownSample %d times to get enough pseudo bulks!" % times)
        for item in bulks:
            cur_index = np.argwhere(key_index == item).reshape(-1)
            index_list.append(cur_index)

            length = cur_index.shape[0]
            for i in range(times):
                shuffle_index = cur_index.copy()
                np.random.shuffle(shuffle_index)
                index_list.append(shuffle_index[0:max(1, int(length * r))])
    else:
        for item in bulks:
            cur_index = np.argwhere(key_index == item).reshape(-1)
            index_list.append(cur_index)
            df_index.append(item)

    for i, index in enumerate(index_list):
        if i == 0:
            data_mtx = np.average(adata_ref[index, :].X.toarray(), axis=0).reshape(1, -1)
        else:
            data_mtx = np.concatenate([data_mtx, np.average(adata_ref[index, :].X.toarray(), axis=0).reshape(1, -1)],
                                      axis=0)
    print("Finished, bulk_data's shape:", data_mtx.shape)
    adata_bulk = sc.AnnData(data_mtx, var=adata_ref.var.copy())

    return adata_bulk

def optim_parameters(net, included=None, excluded=None):
    def belongs_to(cur_layer, layers):
        for layer in layers:
            if layer in cur_layer:
                return True
        return False
    
    params = []
    if included is not None:
        if not isinstance(included, list):
            included = [included]
        for cur_layer, param in net.named_parameters():
            if belongs_to(cur_layer, included) and param.requires_grad:
                params.append(param)
    else:
        if not isinstance(excluded, list):
            excluded = [excluded] if excluded is not None else []
        for cur_layer, param in net.named_parameters():
            if not belongs_to(cur_layer, excluded) and param.requires_grad:
                params.append(param)
    
    return iter(params)

def reads_to_fragments(
    adata: AnnData,
    layer: Optional[str] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
):
    """
    Function to convert read counts to appoximate fragment counts

    Parameters
    ----------
    adata
        AnnData object that contains read counts.
    layer
        Layer that the read counts are stored in.
    key_added
        Name of layer where the fragment counts will be stored.
    copy
        Whether to modify copied input object.
    """
    if copy:
        adata = adata.copy()

    if layer:
        data = np.ceil(adata.layers[layer].data / 2)
    else:
        data = np.ceil(adata.X.data / 2)

    if key_added:
        adata.layers[key_added] = adata.X.copy()
        adata.layers[key_added].data = data
    elif layer and key_added is None:
        adata.layers[layer].data = data
    elif layer is None and key_added is None:
        adata.X.data = data
    if copy:
        return adata

def load_anndata_from_df(df_file):
    df = pd.read_csv(df_file, index_col=0)
    embeddings_col = []
    RNA_col = []
    CAS_col = []
    ADT_col = []
    anno_col = []
    for item in df.columns:
        if item.split("_")[0] in ["PC", "joint", "scAIembeddings", "LSI", "SpatialPCs"]: embeddings_col.append(item)
        elif item.split("_")[0] in ["RNA"]: RNA_col.append(item)
        elif item.split("_")[0] in ["CAS"]: CAS_col.append(item)
        elif item.split("_")[0] in ["ADT"]: ADT_col.append(item)
        else: anno_col.append(item)
    adata = sc.AnnData(df.loc[:, embeddings_col], obs=df.loc[:, anno_col]) if len(embeddings_col)>0 else sc.AnnData(np.ones((df.shape[0], 50)), obs=df.loc[:, anno_col])
    if len(embeddings_col)>=5: adata.obsm["embeddings"] = df.loc[:, embeddings_col].values
    if len(RNA_col)>=5: adata.obsm["RNA_embeddings"] = df.loc[:, RNA_col].values
    if len(CAS_col)>=5: adata.obsm["CAS_embeddings"] = df.loc[:, CAS_col].values
    if len(ADT_col)>=5: adata.obsm["ADT_embeddings"] = df.loc[:, ADT_col].values
        
    return adata

def TFIDF(count_mat, type_=2):
    # Perform TF-IDF (count_mat: peak*cell)
    def tfidf1(count_mat): 
        if not scipy.sparse.issparse(count_mat):
            count_mat = scipy.sparse.coo_matrix(count_mat)

        nfreqs = count_mat.multiply(1.0 / count_mat.sum(axis=0))
        tfidf_mat = nfreqs.multiply(np.log(1 + 1.0 * count_mat.shape[1] / count_mat.sum(axis=1)).reshape(-1,1)).tocoo()

        return scipy.sparse.csr_matrix(tfidf_mat)

    # Perform Signac TF-IDF (count_mat: peak*cell) [default selected]
    def tfidf2(count_mat): 
        if not scipy.sparse.issparse(count_mat):
            count_mat = scipy.sparse.coo_matrix(count_mat)

        tf_mat = count_mat.multiply(1.0 / count_mat.sum(axis=0))
        signac_mat = (1e4 * tf_mat).multiply(1.0 * count_mat.shape[1] / count_mat.sum(axis=1).reshape(-1,1))
        signac_mat = signac_mat.log1p()

        return scipy.sparse.csr_matrix(signac_mat)

    # Perform TF-IDF (count_mat: ?)
    from sklearn.feature_extraction.text import TfidfTransformer
    def tfidf3(count_mat): 
        model = TfidfTransformer(smooth_idf=False, norm="l2")
        model = model.fit(np.transpose(count_mat))
        model.idf_ -= 1
        tf_idf = np.transpose(model.transform(np.transpose(count_mat)))

        return scipy.sparse.csr_matrix(tf_idf)
    
    if type_==1:
        return tfidf1(count_mat)
    elif type_==2:
        return tfidf2(count_mat)
    else:
        return tfidf3(count_mat)

def CLR_transform(ADT_mat):
    """
    Centered log-ratio transformation for ADT matrix.

    Parameters
    ----------
    ADT_mat: sparse or dense matrix
        ADT matrix for processing.

    Returns
    ----------
    ADT_mat_processed: sparse matrix
        ADT matrix with CLR transformation preprocessed.

    gmean_list
        vector of geometric mean for ADT expression of each cell.
    """
    ADT_mat_processed = ADT_mat.todense() if scipy.sparse.issparse(ADT_mat) else ADT_mat.copy()
    gmean_list = []
    for i in range(ADT_mat_processed.shape[0]):
        temp = []
        for j in range(ADT_mat_processed.shape[1]):
            if not ADT_mat_processed[i, j] == 0:
                temp.append(ADT_mat_processed[i, j])
        gmean_temp = gmean(temp)
        gmean_list.append(gmean_temp)
        for j in range(ADT_mat_processed.shape[1]):
            if not ADT_mat_processed[i, j] == 0:
                ADT_mat_processed[i, j] = np.log(ADT_mat_processed[i, j] / gmean_temp)
    ADT_mat_processed = scipy.sparse.csr_matrix(ADT_mat_processed)
    
    return ADT_mat_processed, gmean_list
    
def default_louvain(adata, use_rep="embeddings", key_added="Dlouvain", resolution=1.0):
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.louvain(adata, resolution=resolution)
    adata.obs[key_added] = adata.obs["louvain"]

    return adata

def run_louvain(adata, n_cluster, use_rep="embeddings", key_added="Nlouvain", range_min=0, range_max=3, max_steps=30, tolerance=0):
    sc.pp.neighbors(adata, use_rep=use_rep)
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max-this_min)/2)
        sc.tl.louvain(adata, resolution=this_resolution)
        this_clusters = adata.obs['louvain'].nunique()

        if this_clusters > n_cluster+tolerance:
            this_max = this_resolution
        elif this_clusters < n_cluster-tolerance:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f"%(n_cluster, this_resolution))
            adata.obs[key_added] = adata.obs["louvain"]
            
            return adata
            
        this_step += 1

    print('Cannot find the number of clusters')
    adata.obs[key_added] = adata.obs["louvain"]

    return adata

def default_leiden(adata, use_rep="embeddings", key_added="Dleiden", resolution=1.0):
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.leiden(adata, resolution=resolution)
    adata.obs[key_added] = adata.obs["leiden"]

    return adata

def run_leiden(adata, n_cluster, use_rep="embeddings", key_added="Nleiden", range_min=0, range_max=3, max_steps=30, tolerance=0):
    sc.pp.neighbors(adata, use_rep=use_rep)
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max-this_min)/2)
        sc.tl.leiden(adata, resolution=this_resolution)
        this_clusters = adata.obs['leiden'].nunique()

        if this_clusters > n_cluster+tolerance:
            this_max = this_resolution
        elif this_clusters < n_cluster-tolerance:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f"%(n_cluster, this_resolution))
            adata.obs[key_added] = adata.obs["leiden"]
            
            return adata
        
        this_step += 1
    
    print('Cannot find the number of clusters')
    adata.obs[key_added] = adata.obs["leiden"]
    return adata

def Anndata2Rdatadir(adata, out_dir, spatial=True):
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if scipy.sparse.issparse(adata.X):
        counts = adata.X.tocoo()
    else:
        counts = scipy.sparse.coo_matrix(adata.X)
    counts = pd.DataFrame({"i":counts.row, "j":counts.col, "x":counts.data})
    counts.to_csv(f"{out_dir}/counts.txt", index=True)
    cell_meta = pd.concat([adata.obs, pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names, columns=["X", "Y"])], axis=1) if spatial else adata.obs.copy()
    cell_meta.to_csv(f"{out_dir}/cell_meta.csv")
    gene_meta = adata.var.copy()
    gene_meta.to_csv(f"{out_dir}/gene_meta.csv")
    print("finished")

def setup_seed(seed):
    import torch

    random.seed(seed)                                                            
    torch.manual_seed(seed)                                                      
    torch.cuda.manual_seed_all(seed)                                             
    np.random.seed(seed)                                                         
    os.environ['PYTHONHASHSEED'] = str(seed)                                     
    torch.backends.cudnn.deterministic = True                                    
    torch.backends.cudnn.benchmark = False 
    
def data2input(data):
    import torch

    if scipy.sparse.issparse(data):
        data = data.toarray()
    if not isinstance(data, torch.Tensor):
        data = torch.LongTensor(data) if str(data.dtype).startswith("int") else torch.FloatTensor(data)
        
    return data

def GeometricData(graph, fea_mat=None, node_index=None):
    import torch
    
    from torch_geometric.utils import dense_to_sparse
    from torch_geometric.data import Data
    
    node_feature = data2input(fea_mat) if fea_mat is not None else None
    node_index = data2input(node_index) if node_index is not None else None
    _graph = data2input(graph)
    edge_index, edge_attr = dense_to_sparse(_graph)
    dataset = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr, node_index=node_index)
    
    return dataset

class EarlyStopping(object):
    """Early stops the training if the training loss doesn't decrease after a given patience."""
    def __init__(self, patience=10, delta=1e-4):
        """
        Args:
            patience (int): How long to wait after last time training loss decreased.
                            Default: 5
            delta (float): Minimum change in the monitored quantity as a decrease.
                            Default: 1e-4
        """
        
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
#         self.is_save = False
        
#     def save_model(self, model, save_dir="model_param"):
#         os.makedirs(save_dir, exist_ok=True)
#         torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
#         self.is_save = False
    
#     def load_model_param(self, model, save_dir="model_param"):
#         path = os.path.join(save_dir, "best_model.pth")
#         model.load_state_dict(torch.load(path))
        
    def __call__(self, train_loss):
#         if self.best_loss is None or train_loss < self.best_loss:
#             self.is_save = True
        if self.best_loss is None:
            self.best_loss = train_loss
            self.counter = 0
        elif self.best_loss - train_loss < self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = train_loss
            self.counter = 0
            
def find_peak_overlaps(query, key):
    q_seqname = np.array(query.get_seqnames())
    k_seqname = np.array(key.get_seqnames())
    q_start = np.array(query.get_start())
    k_start = np.array(key.get_start())
    q_width = np.array(query.get_width())
    k_width = np.array(key.get_width())
    q_end = q_start + q_width
    k_end = k_start + k_width
    
    q_index = 0
    k_index = 0
    overlap_index = [[] for i in range(len(query))]
    overlap_count = [0 for i in range(len(query))]
    
    while True:
        if q_index == len(query) or k_index == len(key):
            return overlap_index, overlap_count
        
        if q_seqname[q_index] == k_seqname[k_index]:
            if k_start[k_index] >= q_start[q_index] and k_end[k_index] <= q_end[q_index]:
                overlap_index[q_index].append(k_index)
                overlap_count[q_index] += 1
                k_index += 1
            elif k_start[k_index] < q_start[q_index]:
                k_index += 1
            else:
                q_index += 1
        elif q_seqname[q_index] < k_seqname[k_index]:
            q_index += 1
        else:
            k_index += 1
            
def peak_sets_alignment(adata_list, sep=(":", "-"), min_width=20, max_width=10000, min_gap_width=1, peak_region: Optional[str] = None):
    from genomicranges import GenomicRanges
    from iranges import IRanges
    from biocutils.combine import combine

    ## Peak merging
    gr_list = []
    for i in range(len(adata_list)):
        seq_names = []
        starts = []
        widths = []
        regions = adata_list[i].var_names if peak_region is None else adata_list[i].obs[peak_region]
        for region in regions:
            seq_names.append(region.split(sep[0])[0])
            if sep[0] == sep[1]:
                start, end = region.split(sep[0])[1:]
            else:
                start, end = region.split(sep[0])[1].split(sep[1])
            width = int(end) - int(start)
            starts.append(int(start))
            widths.append(width)
        gr = GenomicRanges(seqnames = seq_names, ranges=IRanges(starts, widths)).sort()
        peaks = [seqname+sep[0]+str(start)+sep[1]+str(end) for seqname, start, end in zip(gr.get_seqnames(), gr.get_start(), gr.get_end())]
        adata_list[i] = adata_list[i][:, peaks]
        gr_list.append(gr)
        
    gr_combined = combine(*gr_list)
    gr_merged = gr_combined.reduce(min_gap_width = min_gap_width).sort()
    print("Peak merged")

    ## Peak filtering
    # filter by intesect
    overlap_index_list = []
    index = np.ones(len(gr_merged)).astype(bool)
    for gr in gr_list:
        overlap_index, overlap_count = find_peak_overlaps(gr_merged, gr)
        index = (np.array(overlap_count) > 0) * index
        overlap_index_list.append(overlap_index)
    # filter by width
    index = index * (gr_merged.get_width() > min_width) * (gr_merged.get_width() < max_width)
    gr_merged = gr_merged.get_subset(index)
    common_peak = [seqname+":"+str(start)+"-"+str(end) for seqname, start, end in zip(gr_merged.get_seqnames(), gr_merged.get_start(), gr_merged.get_end())]
    print("Peak filtered")
    
    ## Merge count matrix
    adata_merged_list = []
    for adata, overlap_index in zip(adata_list, overlap_index_list):
        overlap_index = [overlap_index[i] for i in range(len(index)) if index[i]]
        X = adata.X.tocsc()
        X_merged = scipy.sparse.hstack([scipy.sparse.csr_matrix(X[:, cur].sum(axis=1)) for cur in overlap_index])
        adata_merged_list.append(sc.AnnData(X_merged, obs=adata.obs, var=pd.DataFrame(index=common_peak), obsm=adata.obsm))
    print("Matrix merged")
    
    return adata_merged_list

def gene_sets_alignment(adata_list):
    assert len(adata_list) > 1, "User needs to input more than two datasets"
    for i in range(len(adata_list)):
        adata_list[i].var_names = [item.lower() for item in adata_list[i].var_names]
        adata_list[i].var_names_make_unique()
    common_genes = np.intersect1d(adata_list[0].var_names, adata_list[1].var_names)
    for i in range(2, len(adata_list)):
        common_genes = np.intersect1d(common_genes, adata_list[i].var_names)
    for i in range(len(adata_list)):
        adata_list[i] = adata_list[i][:, common_genes]
    
    return adata_list

def ref_feature_alignment(adata_ref, adata, omics=None):
    """
    Align the feature set of reference dataset with that of the target dataset

    Parameters
    ------
    adata_ref
        reference dataset of anndata format
    adata
        target dataset of anndata format

    Returns
    ------
    scanpy.anndata
        reference dataset aligned with target dataset
    """
    if omics is None or omics.upper() in ["RNA", "ADT"]:
        adata_ref, adata = gene_sets_alignment([adata_ref, adata])
    else:
        adata_ref, adata = peak_sets_alignment([adata_ref, adata])
        
    return adata_ref, adata

def StrLabel2Idx(string_labels):
    # 创建LabelEncoder对象
    label_encoder = LabelEncoder()
    idx_labels = label_encoder.fit_transform(string_labels)
    
    return np.array(idx_labels)

def knn_label_translation(reference_X, reference_y, target_X, k=5):
    label_encoder = LabelEncoder()
    reference_y_idx = label_encoder.fit_transform(reference_y)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(reference_X, reference_y_idx)
    target_y_idx = neigh.predict(target_X)
    target_y = label_encoder.inverse_transform(target_y_idx)

    return target_y

def Integrated_3D_graph(batch_label, spatial_mat, joint_mat=None, rna_mat=None, cas_mat=None, adt_mat=None,
                        intra_neighbors=6, intra_metric="euclidean", inter_neighbors=6, inter_metric="cosine"):
    def find_intra_neighbor(matrix, k, metric='euclidean'):
        k = min(matrix.shape[0]-1, k)
        if k>0:
            nn_model = NearestNeighbors(n_neighbors=k, algorithm='auto', metric=metric)
            nn_model.fit(matrix)
            _, indices = nn_model.kneighbors(matrix, n_neighbors=k+1)
            return indices[:, 1:]
        else:
            return None
    
    def find_inter_neighbor(matrix1, matrix2, k, metric='cosine'):
        if matrix1.shape[1] != matrix2.shape[1]:
            raise ValueError("invalid feature dimension")
        k = min(matrix2.shape[0], k)
        if k>0:
            nn_model = NearestNeighbors(n_neighbors=k, algorithm='auto', metric=metric)
            nn_model.fit(matrix2)
            dist, indices = nn_model.kneighbors(matrix1, n_neighbors=k)
            return indices
        else:
            return None
        
    ## Sparse matrix row, col, data initialization
    row_index = []
    col_index = []
    data = []

    N = spatial_mat.shape[0]
    if not isinstance(batch_label, np.ndarray): batch_label = np.array(batch_label)
    samples_indices = np.arange(N)
    for cur in np.unique(batch_label):
        cur_idx = batch_label==cur
        cur_original_idx = samples_indices[cur_idx]
        cur_spatial_mat = spatial_mat[cur_idx]
        indices = find_intra_neighbor(cur_spatial_mat, k=intra_neighbors, metric=intra_metric)
        intra_indices = cur_original_idx[indices] if indices is not None else None
        
        other_idx = batch_label!=cur
        other_original_idx = samples_indices[other_idx]
        inter_indices = []
        if joint_mat is not None:
            if scipy.sparse.issparse(joint_mat): joint_mat = joint_mat.A
            cur_joint_mat = joint_mat[cur_idx]
            other_joint_mat = joint_mat[other_idx]
            indices = find_inter_neighbor(cur_joint_mat, other_joint_mat, k=inter_neighbors, metric=inter_metric)
            if indices is not None: inter_indices.append(other_original_idx[indices])
        if rna_mat is not None:
            if scipy.sparse.issparse(rna_mat): rna_mat = rna_mat.A
            cur_rna_mat = rna_mat[cur_idx]
            other_rna_mat = rna_mat[other_idx]
            indices = find_inter_neighbor(cur_rna_mat, other_rna_mat, k=inter_neighbors, metric=inter_metric)
            if indices is not None: inter_indices.append(other_original_idx[indices])
        if cas_mat is not None:
            if scipy.sparse.issparse(cas_mat): cas_mat = cas_mat.A
            cur_cas_mat = cas_mat[cur_idx]
            other_cas_mat = cas_mat[other_idx]
            indices = find_inter_neighbor(cur_cas_mat, other_cas_mat, k=inter_neighbors, metric=inter_metric)
            if indices is not None: inter_indices.append(other_original_idx[indices])
        if adt_mat is not None:
            if scipy.sparse.issparse(adt_mat): adt_mat = adt_mat.A
            cur_adt_mat = adt_mat[cur_idx]
            other_adt_mat = adt_mat[other_idx]
            indices = find_inter_neighbor(cur_adt_mat, other_adt_mat, k=inter_neighbors, metric=inter_metric)
            if indices is not None: inter_indices.append(other_original_idx[indices])
        
        inter_indices = np.concatenate(inter_indices, axis=1) if len(inter_indices) > 0 else None
        for i in range(cur_spatial_mat.shape[0]):
            if intra_indices is not None:
                cur_list = intra_indices[i].tolist()
                row_index += cur_list
                col_index += [cur_original_idx[i]] * len(cur_list)
                data += [1] * len(cur_list)
            if inter_indices is not None:
                cur_list = list(set(inter_indices[i].tolist()))
                row_index += cur_list
                col_index += [cur_original_idx[i]] * len(cur_list)
                data += [-1] * len(cur_list)

    return scipy.sparse.coo_matrix((data, (row_index, col_index)), shape=(N, N))