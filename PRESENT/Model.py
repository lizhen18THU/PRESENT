from .Utils import *
from .Layers import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import torch_geometric
from torch_geometric.loader import ClusterData, ClusterLoader

class PRESENT_RP(nn.Module):
    def __init__(
        self, 
        rna_dim: Optional[int] = None, 
        cas_dim: Optional[int] = None,
        adt_dim: Optional[int] = None,
        d_lat: int = 50,
        d_hid: tuple = (1024, 512),
        rna_zero_inflaten: bool = True,
        cas_zero_inflaten: bool = True,
        is_recons: bool = True,
        k_neighbors: int = 6,
        tau: float = 0.1,
        dec_basic_module: str = "Linear",
    ):
        
        super().__init__()
        
        self.is_rna = rna_dim is not None
        self.is_cas = cas_dim is not None
        self.is_adt = adt_dim is not None
        self.num_modalities = self.is_adt+self.is_rna+self.is_cas
        self.rna_dim = rna_dim
        self.cas_dim = cas_dim
        self.adt_dim = adt_dim
        self.d_lat = d_lat
        self.d_prior = d_lat//3
        self.d_lat_adt = d_lat_adt = min(d_lat, adt_dim) if adt_dim is not None else 0
        self.d_prior_adt = d_lat_adt//3 if adt_dim is not None else 0
        self.tau = tau
        self.k_neighbors = k_neighbors
        self.is_recons = is_recons
        self.dec_basic_module = dec_basic_module
        
        assert self.num_modalities > 0, "number of omics layer should be larger than 0"
        assert k_neighbors > 0, "k_neighbors should be larger than 0"
        
        # Encoder
        self.rna_encoder = BayesianGATEncoder(rna_dim, d_lat, d_hid=d_hid) if self.is_rna else None
        self.cas_encoder = BayesianGATEncoder(cas_dim, d_lat, d_hid=d_hid) if self.is_cas else None
        self.adt_encoder = BayesianGATEncoder(adt_dim, d_lat_adt, d_hid=(2*adt_dim//3 + d_lat_adt// 3 , adt_dim//3 + 2*d_lat_adt//3)) if self.is_adt else None
        
        ## Feature fusion
        self.fusion_layer = MLP_Module(int(self.is_rna + self.is_cas) * d_lat + d_lat_adt, (int(self.is_rna + self.is_cas) * d_lat + d_lat_adt, d_lat), d_lat) if self.num_modalities > 1 else None
        
        if self.num_modalities==1 and self.is_adt: d_lat=d_lat_adt
        # Decoder
        self.rna_decoder = ZINBDecoder(d_lat, (d_hid[1], d_hid[0]), rna_dim, zero_inflaten=rna_zero_inflaten, is_recons=self.is_recons, basic_module=dec_basic_module) if self.is_rna else None
        self.cas_decoder = ZIPDecoder(d_lat, (d_hid[1], d_hid[0]), cas_dim, zero_inflaten=cas_zero_inflaten, is_recons=self.is_recons, basic_module=dec_basic_module) if self.is_cas else None
        self.adt_decoder = MLP_Module(d_lat, (adt_dim//3 + 2*d_lat_adt//3, 2*adt_dim//3 + d_lat_adt// 3), adt_dim) if self.is_adt else None
        
    def prior_initialize(self, prior_rna, prior_cas, prior_adt, tight_factor=10):
        if self.is_rna: self.rna_encoder.prior_initialize(prior_rna, tight_factor)
        if self.is_cas: self.cas_encoder.prior_initialize(prior_cas, tight_factor)
        if self.is_adt: self.adt_encoder.prior_initialize(prior_adt, tight_factor)
        
    def bnn_loss(self):
        bnnloss = 0
        if self.is_rna: bnnloss += self.rna_encoder.bnn_loss()
        if self.is_cas: bnnloss += self.cas_encoder.bnn_loss()
        if self.is_adt: bnnloss += self.adt_encoder.bnn_loss()
        
        return bnnloss
    
    def freeze(self):
        if self.is_rna: self.rna_encoder.freeze()
        if self.is_cas: self.cas_encoder.freeze()
        if self.is_adt: self.adt_encoder.freeze()
        
    def unfreeze(self):
        if self.is_rna: self.rna_encoder.unfreeze()
        if self.is_cas: self.cas_encoder.unfreeze()
        if self.is_adt: self.adt_encoder.unfreeze()
    
    def forward_encoder(self, rna, cas, adt, edge_index):
        out = []
        if self.is_rna: 
            rna = self.rna_encoder(rna, edge_index)
            out.append(rna)
        if self.is_cas: 
            cas = self.cas_encoder(cas, edge_index)
            out.append(cas)
        if self.is_adt: 
            adt = self.adt_encoder(adt, edge_index)
            out.append(adt)
        if self.num_modalities > 1:
            out = torch.cat(out, -1)
            out = self.fusion_layer(out)
        else:
            out = out[0]
        
        return out, rna, cas, adt
    
    def forward_decoder(self, x_lat, edge_index=None):
        rna = cas = adt = None
        if self.is_rna:
            rna_pi, rna_disp, rna_mean, rna_recons = self.rna_decoder(x_lat, edge_index)
            rna = (rna_pi, rna_disp, rna_mean, rna_recons)
        if self.is_cas:
            cas_pi, cas_omega, _, cas_recons = self.cas_decoder(x_lat, edge_index)
            cas = (cas_pi, cas_omega, _, cas_recons)
        if self.is_adt:
            adt = self.adt_decoder(x_lat)
            
        return rna, cas, adt
    
    def forward(self, rna_norm, rna_counts, rna_libsize, 
                cas_norm, cas_counts, cas_libsize,
                adt_norm, edge_index,
                rna_ridge_lambda=0.5, cas_ridge_lambda=0.5):
        x_lat, rna_lat, cas_lat, adt_lat = self.forward_encoder(rna_norm, cas_norm, adt_norm, edge_index)
        
        rna_ioa_data = rna_lat.data if self.is_rna else None
        cas_ioa_data = cas_lat.data if self.is_cas else None
        adt_ioa_data = adt_lat.data if self.is_adt else None
        ioa_loss = IOA_loss(rna_ioa_data, cas_ioa_data, adt_ioa_data, x_lat, tau=self.tau) if self.num_modalities > 1 else None
        
        rna, cas, adt = self.forward_decoder(x_lat) if self.dec_basic_module == "Linear" else self.forward_decoder(x_lat, edge_index)
        nll_loss = 0
        mse_loss = 0
        if self.is_rna: 
            nll_loss += NLL_loss(rna_counts, rna[0], rna[1], rna[2], scale_factor=rna_libsize, ridge_lambda=rna_ridge_lambda)
            if self.is_recons: mse_loss += F.mse_loss(rna[3], rna_norm)
        if self.is_cas: 
            nll_loss += NLL_loss(cas_counts, cas[0], cas[1], cas[2], scale_factor=cas_libsize, ridge_lambda=cas_ridge_lambda)
            if self.is_recons: mse_loss += F.mse_loss(cas[3], cas_norm)
        if self.is_adt:
            mse_loss += F.mse_loss(adt, adt_norm)
        
        return ioa_loss, nll_loss, mse_loss
    
    def inner_inference(self, edge_index, rna_norm=None, cas_norm=None, adt_norm=None, impute=True, device=torch.device("cpu")):
        self.to(device)
        self.eval()
        
        rna_norm = data2input(rna_norm).to(device) if rna_norm is not None else None
        cas_norm = data2input(cas_norm).to(device) if cas_norm is not None else None
        adt_norm = data2input(adt_norm).to(device) if adt_norm is not None else None
        edge_index = data2input(edge_index).to(device)
        
        rna_lat = cas_lat = adt_lat = None
        rna_imputed = cas_imputed = adt_imputed = None
        with torch.no_grad():
            x_lat, rna_lat, cas_lat, adt_lat = self.forward_encoder(rna_norm, cas_norm, adt_norm, edge_index)
            if impute: rna, cas, adt = self.forward_decoder(x_lat) if self.dec_basic_module == "Linear" else self.forward_decoder(x_lat, edge_index)
            
        x_lat = x_lat.cpu().numpy()
        if self.is_rna: 
            rna_lat = rna_lat.cpu().numpy()
            if impute: rna_imputed = rna[2].cpu().numpy()
        if self.is_cas: 
            cas_lat = cas_lat.cpu().numpy()
            if impute: cas_imputed = cas[2].cpu().numpy()
        if self.is_adt: 
            adt_lat = adt_lat.cpu().numpy()
            if impute: adt_imputed = adt.cpu().numpy()
        
        return x_lat, (rna_lat, cas_lat, adt_lat), (rna_imputed, cas_imputed, adt_imputed)
    
    def model_train(self, spa_mat, rna_counts=None, cas_counts=None, adt_counts=None, impute=False,
                    ref_rna_counts=None, ref_rna_anno=None, ref_cas_counts=None, ref_cas_anno=None, ref_adt_counts=None, ref_adt_anno=None,
                    epochs=100, lr=1e-3, weight_decay=1e-4, batch_size=320, patience=20, delta=1e-4, device=torch.device("cuda:0"), 
                    rna_ridge_lambda=0.5, cas_ridge_lambda=0.5, alpha1=1.0, alpha2=1.0, alpha3=1.0, alpha4=1.0):
        if self.is_rna:
            assert rna_counts is not None and rna_counts.shape[1] == self.rna_dim, "Invalid shape of input rna_counts"
            rna_counts = scipy.sparse.csr_matrix(rna_counts) if not scipy.sparse.issparse(rna_counts) else rna_counts
            if ref_rna_counts is not None:
                assert ref_rna_counts.shape[1] == self.rna_dim, "Invalid shape of input ref_rna_counts"
                ref_rna_counts = scipy.sparse.csr_matrix(ref_rna_counts) if not scipy.sparse.issparse(ref_rna_counts) else ref_rna_counts
        if self.is_cas:
            assert cas_counts is not None and cas_counts.shape[1] == self.cas_dim, "Invalid shape of input cas_counts"
            cas_counts = scipy.sparse.csr_matrix(cas_counts) if not scipy.sparse.issparse(cas_counts) else cas_counts
            if ref_cas_counts is not None:
                assert ref_cas_counts.shape[1] == self.cas_dim, "Invalid shape of input ref_cas_counts"
                ref_cas_counts = scipy.sparse.csr_matrix(ref_cas_counts) if not scipy.sparse.issparse(ref_cas_counts) else ref_cas_counts
        if self.is_adt:
            assert adt_counts is not None and adt_counts.shape[1] == self.adt_dim, "Invalid shape of input adt_counts"
            adt_counts = scipy.sparse.csr_matrix(adt_counts) if not scipy.sparse.issparse(adt_counts) else adt_counts
            if ref_adt_counts is not None:
                assert ref_adt_counts.shape[1] == self.adt_dim, "Invalid shape of input ref_adt_counts"
                ref_adt_counts = scipy.sparse.csr_matrix(ref_adt_counts) if not scipy.sparse.issparse(ref_adt_counts) else ref_adt_counts

        # normalization and get prior weight
        if self.is_rna:
            sdata = sc.AnnData(rna_counts.copy())
            sc.pp.normalize_total(sdata)
            sc.pp.log1p(sdata)
            if ref_rna_counts is not None:
                rdata = sc.AnnData(ref_rna_counts.copy())
                sc.pp.normalize_total(rdata)
                sc.pp.log1p(rdata)
                if ref_rna_anno is not None:
                    rdata.obs["anno"] = np.array(ref_rna_anno)
                    rdata = construct_pseudo_bulk(rdata, key="anno", min_samples=self.d_prior)
                sc.tl.pca(rdata, n_comps=self.d_prior)
                rna_prior = torch.FloatTensor(rdata.varm["PCs"].T.copy())
            else:
                sc.tl.pca(sdata, n_comps=self.d_prior)
                rna_prior = torch.FloatTensor(sdata.varm["PCs"].T.copy())
            rna_norm = sdata.X.copy()
            rna_libsize = (rna_counts.A.sum(axis=1) / np.median(rna_counts.A.sum(axis=1))).reshape(-1, 1)
        else:
            rna_prior = rna_norm = rna_libsize = None
        if self.is_cas:
            sdata = sc.AnnData(cas_counts.copy())
            sdata.X = TFIDF(sdata.X.T).T.copy()
            if ref_cas_counts is not None:
                rdata = sc.AnnData(ref_cas_counts.copy())
                rdata.X = TFIDF(rdata.X.T).T.copy()
                if ref_cas_anno is not None:
                    rdata.obs["anno"] = np.array(ref_cas_anno)
                    rdata = construct_pseudo_bulk(rdata, key="anno", min_samples=self.d_prior)
                sc.tl.pca(rdata, n_comps=self.d_prior)
                cas_prior = torch.FloatTensor(rdata.varm["PCs"].T.copy())
            else:
                sc.tl.pca(sdata, n_comps=self.d_prior)
                cas_prior = torch.FloatTensor(sdata.varm["PCs"].T.copy())
            cas_norm = sdata.X.copy()
            cas_libsize = cas_counts.A.sum(axis=1).reshape(-1, 1)
        else:
            cas_prior = cas_norm = cas_libsize = None
        if self.is_adt:
            sdata = sc.AnnData(adt_counts.copy())
            X_clred, gmean_list = CLR_transform(sdata.X)
            sdata.X = X_clred
            sc.pp.scale(sdata)
            if ref_adt_counts is not None:
                rdata = sc.AnnData(ref_adt_counts.copy())
                X_clred, gmean_list = CLR_transform(rdata.X)
                rdata.X = X_clred
                sc.pp.scale(rdata)
                if ref_adt_anno is not None:
                    rdata.obs["anno"] = np.array(ref_adt_anno)
                    rdata = construct_pseudo_bulk(rdata, key="anno", min_samples=self.d_prior_adt)
                sc.tl.pca(rdata, n_comps=self.d_prior_adt)
                adt_prior = torch.FloatTensor(rdata.varm["PCs"].T.copy())
            else:
                sc.tl.pca(sdata, n_comps=self.d_prior_adt)
                adt_prior = torch.FloatTensor(sdata.varm["PCs"].T.copy())
            adt_norm = sdata.X.copy()
        else:
            adt_prior = adt_norm = None
        
        self.prior_initialize(rna_prior, cas_prior, adt_prior)
        self.unfreeze()
        self.to(device)
        
        # construct kNN graph with spatial_mtx
        knn_graph = kneighbors_graph(np.array(spa_mat), n_neighbors=self.k_neighbors, include_self=False).T.copy()
        geo_dataset = GeometricData(knn_graph, node_index=torch.arange(spa_mat.shape[0]))
        transform = torch_geometric.transforms.ToUndirected()
        undirected_geo = transform(geo_dataset)
        cluster_data = ClusterData(undirected_geo, num_parts=int(np.ceil(undirected_geo.num_nodes/batch_size)) * 10, recursive=False)
        train_loader = ClusterLoader(cluster_data, batch_size=10, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        pbar = tqdm(range(epochs))
        pbar.set_description('Model training')
        early_stopping = EarlyStopping(patience=patience, delta=delta)
        for epoch in pbar:
            self.train()
            train_loss_list = []
            IOA_loss_list = []
            BNN_loss_list = []
            NLL_loss_list = []
            MSE_loss_list = []
            for cur in train_loader:
                np_index = cur.node_index.numpy()
                edge_index = cur.edge_index.to(device)
                if self.is_rna:
                    rna_counts_batch = data2input(rna_counts[np_index]).to(device)
                    rna_norm_batch = data2input(rna_norm[np_index]).to(device)
                    rna_libsize_batch = data2input(rna_libsize[np_index]).to(device)
                else:
                    rna_counts_batch = rna_norm_batch = rna_libsize_batch = None
                if self.is_cas:
                    cas_counts_batch = data2input(cas_counts[np_index]).to(device)
                    cas_norm_batch = data2input(cas_norm[np_index]).to(device)
                    cas_libsize_batch = data2input(cas_libsize[np_index]).to(device)
                else:
                    cas_counts_batch = cas_norm_batch = cas_libsize_batch = None
                if self.is_adt:
                    adt_norm_batch = data2input(adt_norm[np_index]).to(device)
                else:
                    adt_norm_batch = None
                
                ioa_loss, nll_loss, mse_loss = self.forward(rna_norm_batch, rna_counts_batch, rna_libsize_batch,
                                                            cas_norm_batch, cas_counts_batch, cas_libsize_batch,
                                                            adt_norm_batch, edge_index, 
                                                            rna_ridge_lambda=rna_ridge_lambda, cas_ridge_lambda=cas_ridge_lambda)
                bnn_loss = self.bnn_loss()
                if self.num_modalities > 1 and self.is_recons:
                    loss = alpha1*bnn_loss + alpha2*ioa_loss + alpha3*nll_loss + alpha4*mse_loss
                elif self.is_recons:
                    loss = alpha1*bnn_loss + alpha3*nll_loss + alpha4*mse_loss
                elif self.num_modalities > 1:
                    loss = alpha1*bnn_loss + alpha2*ioa_loss + alpha3*nll_loss
                else:
                    loss = alpha1*bnn_loss + alpha3*nll_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())
                if not isinstance(nll_loss, int): NLL_loss_list.append(nll_loss.item())
                else: NLL_loss_list.append(nll_loss)
                BNN_loss_list.append(bnn_loss.item())
                if self.num_modalities > 1: IOA_loss_list.append(ioa_loss.item())
                if self.is_recons: MSE_loss_list.append(mse_loss.item())
            train_loss = np.mean(train_loss_list)
            nll_loss = np.mean(NLL_loss_list)
            bnn_loss = np.mean(BNN_loss_list)
            early_stopping(train_loss)
            if self.num_modalities > 1 and self.is_recons:
                ioa_loss = np.mean(IOA_loss_list)
                mse_loss = np.mean(MSE_loss_list)
                pbar.set_postfix({'NLL_loss': nll_loss, 'BNN_loss': bnn_loss, 'MSE_loss': mse_loss, 'IOA_loss': ioa_loss, 'ES counter':early_stopping.counter, "ES patience": early_stopping.patience})
            elif self.is_recons:
                mse_loss = np.mean(MSE_loss_list)
                pbar.set_postfix({'NLL_loss': nll_loss, 'BNN_loss': bnn_loss, 'MSE_loss': mse_loss, 'ES counter':early_stopping.counter, "ES patience": early_stopping.patience})
            elif self.num_modalities > 1:
                ioa_loss = np.mean(IOA_loss_list)
                pbar.set_postfix({'NLL_loss': nll_loss, 'BNN_loss': bnn_loss, 'IOA_loss': ioa_loss, 'ES counter':early_stopping.counter, "ES patience": early_stopping.patience})
            else:
                pbar.set_postfix({'NLL_loss': nll_loss, 'BNN_loss': bnn_loss, 'ES counter':early_stopping.counter, "ES patience": early_stopping.patience})
            if early_stopping.early_stop:
                print("Early stop the training process")
                break
        self.freeze()
        x_lat, omics_lat, omics_imputed = self.inner_inference(undirected_geo.edge_index, rna_norm, cas_norm, adt_norm, impute=impute)
        
        return x_lat, omics_lat, omics_imputed

class PRESENT_BC(nn.Module):
    def __init__(
        self, 
        rna_dim: Optional[int] = None, 
        cas_dim: Optional[int] = None,
        adt_dim: Optional[int] = None,
        n_batches: Optional[int] = None,
        d_lat: int = 50,
        d_hid: tuple = (1024, 512),
        rna_zero_inflaten: bool = True,
        cas_zero_inflaten: bool = True,
        is_recons: bool = True,
        intra_neighbors: int = 6,
        inter_neighbors: int = 6,
        tau: float = 0.1):
        
        super().__init__()
        
        if n_batches is None or n_batches==1:
            n_batches = None
            self.n_batches = None
        else:
            self.n_batches = n_batches
        self.is_rna = rna_dim is not None
        self.is_cas = cas_dim is not None
        self.is_adt = adt_dim is not None
        self.is_recons = is_recons,
        self.num_modalities = self.is_adt + self.is_rna + self.is_cas
        assert self.n_batches is not None and self.n_batches > 1, "Input batches should be larger than 1"
        assert self.num_modalities > 0, "Input modalities should be larger than 0"
        self.rna_dim = rna_dim
        self.cas_dim = cas_dim
        self.adt_dim = adt_dim
        self.d_lat = d_lat
        self.d_prior = d_lat//3
        self.d_lat_adt = d_lat_adt = min(d_lat, adt_dim) if adt_dim is not None else 0
        self.d_prior_adt = d_lat_adt//3 if adt_dim is not None else 0
        self.tau = tau
        self.intra_neighbors = intra_neighbors
        self.inter_neighbors = inter_neighbors
        
        # Encoder
        self.rna_encoder = BayesianGATEncoder(rna_dim, d_lat, d_hid=d_hid) if self.is_rna else None
        self.cas_encoder = BayesianGATEncoder(cas_dim, d_lat, d_hid=d_hid) if self.is_cas else None
        self.adt_encoder = BayesianGATEncoder(adt_dim, d_lat_adt, d_hid=(2*adt_dim//3 + d_lat_adt// 3 , adt_dim//3 + 2*d_lat_adt//3)) if self.is_adt else None
        
        ## Feature fusion
        self.fusion_layer = MLP_Module(int(self.is_rna + self.is_cas) * d_lat + d_lat_adt, (int(self.is_rna + self.is_cas) * d_lat + d_lat_adt, d_lat), d_lat)
        self.batch_discriminator = MLP_Module(d_lat, (d_lat, d_lat), d_lat)
        
        if self.num_modalities==1 and self.is_adt: d_lat=d_lat_adt
        # Decoder
        self.register_parameter('batch_embeddings', nn.Parameter(torch.eye(n_batches), requires_grad=False))
        d_lat = d_lat + n_batches
        self.rna_decoder = ZINBDecoder(d_lat, (d_hid[1], d_hid[0]), rna_dim, zero_inflaten=rna_zero_inflaten, is_recons=is_recons) if self.is_rna else None
        self.cas_decoder = ZIPDecoder(d_lat, (d_hid[1], d_hid[0]), cas_dim, zero_inflaten=cas_zero_inflaten, is_recons=is_recons) if self.is_cas else None
        self.adt_decoder = MLP_Module(d_lat, (adt_dim//3 + 2*d_lat_adt//3, 2*adt_dim//3 + d_lat_adt// 3), adt_dim) if self.is_adt else None
            
    def prior_initialize(self, prior_rna, prior_cas, prior_adt, tight_factor=10):
        if self.is_rna: self.rna_encoder.prior_initialize(prior_rna, tight_factor)
        if self.is_cas: self.cas_encoder.prior_initialize(prior_cas, tight_factor)
        if self.is_adt: self.adt_encoder.prior_initialize(prior_adt, tight_factor)
        
    def bnn_loss(self):
        bnnloss = 0
        if self.is_rna: bnnloss += self.rna_encoder.bnn_loss()
        if self.is_cas: bnnloss += self.cas_encoder.bnn_loss()
        if self.is_adt: bnnloss += self.adt_encoder.bnn_loss()
        return bnnloss
    
    def freeze(self):
        if self.is_rna: self.rna_encoder.freeze()
        if self.is_cas: self.cas_encoder.freeze()
        if self.is_adt: self.adt_encoder.freeze()
        
    def unfreeze(self):
        if self.is_rna: self.rna_encoder.unfreeze()
        if self.is_cas: self.cas_encoder.unfreeze()
        if self.is_adt: self.adt_encoder.unfreeze()
    
    def forward_encoder(self, rna, cas, adt, edge_index):
        out = []
        if self.is_rna: 
            rna = self.rna_encoder(rna, edge_index)
            out.append(rna)
        if self.is_cas: 
            cas = self.cas_encoder(cas, edge_index)
            out.append(cas)
        if self.is_adt: 
            adt = self.adt_encoder(adt, edge_index)
            out.append(adt)
        if self.num_modalities > 1:
            out = torch.cat(out, -1)
            out = self.fusion_layer(out)
        else:
            out = out[0]
        
        return out, rna, cas, adt
    
    def forward_decoder(self, x_lat, batch_indices):
        x_lat = torch.cat([x_lat, self.batch_embeddings[batch_indices]], -1)
        rna = cas = adt = None
        if self.is_rna:
            rna_pi, rna_disp, rna_mean, rna_recons = self.rna_decoder(x_lat)
            rna = (rna_pi, rna_disp, rna_mean, rna_recons)
        if self.is_cas:
            cas_pi, cas_omega, _, cas_recons = self.cas_decoder(x_lat)
            cas = (cas_pi, cas_omega, _, cas_recons)
        if self.is_adt:
            adt = self.adt_decoder(x_lat)
            
        return rna, cas, adt
    
    def forward(self, rna_norm, rna_counts, rna_libsize, 
                cas_norm, cas_counts, cas_libsize,
                adt_norm, edge_index,
                batch_indices, positive_indices=None, negative_indices=None, target_lat=None,
                rna_ridge_lambda=0.5, cas_ridge_lambda=0.5, stage=1):
        x_lat, rna_lat, cas_lat, adt_lat = self.forward_encoder(rna_norm, cas_norm, adt_norm, edge_index)
        if self.num_modalities == 1 and stage==2: x_lat = self.fusion_layer(x_lat)
        iba_loss = IBA_loss(x_lat, positive_indices, negative_indices, tau=self.tau) if positive_indices is not None and negative_indices is not None else 0
        ibp_loss = IBP_loss(x_lat, target_lat, batch_indices, tau=self.tau) if target_lat is not None else 0
        
        rna_ioa_data = rna_lat.data if self.is_rna else None
        cas_ioa_data = cas_lat.data if self.is_cas else None
        adt_ioa_data = adt_lat.data if self.is_adt else None
        ioa_loss = IOA_loss(rna_ioa_data, cas_ioa_data, adt_ioa_data, x_lat, tau=self.tau) if self.num_modalities > 1 else 0
        
        rna, cas, adt = self.forward_decoder(x_lat, batch_indices)
        nll_loss = 0
        mse_loss = 0
        if self.is_rna: 
            nll_loss += NLL_loss(rna_counts, rna[0], rna[1], rna[2], scale_factor=rna_libsize, ridge_lambda=rna_ridge_lambda)
            if self.is_recons: mse_loss += F.mse_loss(rna[3], rna_norm)
        if self.is_cas: 
            nll_loss += NLL_loss(cas_counts, cas[0], cas[1], cas[2], scale_factor=cas_libsize, ridge_lambda=cas_ridge_lambda)
            if self.is_recons: mse_loss += F.mse_loss(cas[3], cas_norm)
        if self.is_adt: 
            mse_loss += F.mse_loss(adt, adt_norm)
            
        return x_lat, nll_loss, mse_loss, ioa_loss, iba_loss, ibp_loss
    
    def inner_inference(self, edge_index, rna_norm=None, cas_norm=None, adt_norm=None, batch_indices=None, impute=True, batch_fusion=True, device=torch.device("cpu")):
        self.to(device)
        self.eval()
        rna_norm = data2input(rna_norm).to(device) if self.is_rna and rna_norm is not None else None
        cas_norm = data2input(cas_norm).to(device) if self.is_cas and cas_norm is not None else None
        adt_norm = data2input(adt_norm).to(device) if self.is_adt and adt_norm is not None else None
        batch_indices = data2input(batch_indices).to(device) if batch_indices is not None else None
        edge_index = edge_index.to(device)
        
        rna_lat = cas_lat = adt_lat = None
        rna_imputed = cas_imputed = adt_imputed = None
        with torch.no_grad():
            x_lat, rna_lat, cas_lat, adt_lat = self.forward_encoder(rna_norm, cas_norm, adt_norm, edge_index)
            if self.num_modalities == 1 and batch_fusion: x_lat =self.fusion_layer(x_lat)
            if impute: rna, cas, adt = self.forward_decoder(x_lat, batch_indices)
            
        x_lat = x_lat.cpu().numpy()
        if self.is_rna: 
            rna_lat = rna_lat.cpu().numpy()
            if impute: rna_imputed = rna[2].cpu().numpy()
        if self.is_cas: 
            cas_lat = cas_lat.cpu().numpy()
            if impute: cas_imputed = cas[2].cpu().numpy()
        if self.is_adt: 
            adt_lat = adt_lat.cpu().numpy()
            if impute: adt_imputed = adt.cpu().numpy()
        
        return x_lat, (rna_lat, cas_lat, adt_lat), (rna_imputed, cas_imputed, adt_imputed)
    
    def model_train(self, spa_mat, rna_counts=None, cas_counts=None, adt_counts=None, batch_label=None, impute=False,
                    ref_rna_counts=None, ref_rna_anno=None, ref_cas_counts=None, ref_cas_anno=None, ref_adt_counts=None, ref_adt_anno=None,
                    negative_samples=0.25, rna_ridge_lambda=0.5, cas_ridge_lambda=0.5, 
                    epochs=100, lr=1e-3, weight_decay=1e-4, batch_size=320, patience=20, delta=1e-4,
                    alpha1=1.0, alpha2=1.0, alpha3=1.0, alpha4=1.0, beta1=1.0, beta2=1.0, beta3=1.0, device=torch.device("cuda:0")):
        if self.is_rna:
            assert rna_counts is not None and rna_counts.shape[1] == self.rna_dim, "Invalid shape of input rna_counts"
            rna_counts = scipy.sparse.csr_matrix(rna_counts) if not scipy.sparse.issparse(rna_counts) else rna_counts
            if ref_rna_counts is not None:
                assert ref_rna_counts.shape[1] == self.rna_dim, "Invalid shape of input ref_rna_counts"
                ref_rna_counts = scipy.sparse.csr_matrix(ref_rna_counts) if not scipy.sparse.issparse(ref_rna_counts) else ref_rna_counts
        if self.is_cas:
            assert cas_counts is not None and cas_counts.shape[1] == self.cas_dim, "Invalid shape of input cas_counts"
            cas_counts = scipy.sparse.csr_matrix(cas_counts) if not scipy.sparse.issparse(cas_counts) else cas_counts
            if ref_cas_counts is not None:
                assert ref_cas_counts.shape[1] == self.cas_dim, "Invalid shape of input ref_cas_counts"
                ref_cas_counts = scipy.sparse.csr_matrix(ref_cas_counts) if not scipy.sparse.issparse(ref_cas_counts) else ref_cas_counts
        if self.is_adt:
            assert adt_counts is not None and adt_counts.shape[1] == self.adt_dim, "Invalid shape of input adt_counts"
            adt_counts = scipy.sparse.csr_matrix(adt_counts) if not scipy.sparse.issparse(adt_counts) else adt_counts
            if ref_adt_counts is not None:
                assert ref_adt_counts.shape[1] == self.adt_dim, "Invalid shape of input ref_adt_counts"
                ref_adt_counts = scipy.sparse.csr_matrix(ref_adt_counts) if not scipy.sparse.issparse(ref_adt_counts) else ref_adt_counts
        assert batch_label is not None and np.unique(batch_label).shape[0] == self.n_batches, "Invalid input batch_label"
        
        # normalization and get prior weight
        if self.is_rna:
            sdata = sc.AnnData(rna_counts.copy())
            sc.pp.normalize_total(sdata)
            sc.pp.log1p(sdata)
            if ref_rna_counts is not None:
                rdata = sc.AnnData(ref_rna_counts.copy())
                sc.pp.normalize_total(rdata)
                sc.pp.log1p(rdata)
                if ref_rna_anno is not None:
                    rdata.obs["anno"] = np.array(ref_rna_anno)
                    rdata = construct_pseudo_bulk(rdata, key="anno", min_samples=self.d_prior)
                sc.tl.pca(rdata, n_comps=self.d_prior)
                rna_prior = torch.FloatTensor(rdata.varm["PCs"].T.copy())
            else:
                sc.tl.pca(sdata, n_comps=self.d_prior)
                rna_prior = torch.FloatTensor(sdata.varm["PCs"].T.copy())
            rna_norm = sdata.X.copy()
            rna_libsize = (rna_counts.A.sum(axis=1) / np.median(rna_counts.A.sum(axis=1))).reshape(-1, 1)
        else:
            rna_prior = rna_norm = rna_libsize = None
        if self.is_cas:
            sdata = sc.AnnData(cas_counts.copy())
            sdata.X = TFIDF(sdata.X.T).T.copy()
            if ref_cas_counts is not None:
                rdata = sc.AnnData(ref_cas_counts.copy())
                rdata.X = TFIDF(rdata.X.T).T.copy()
                if ref_cas_anno is not None:
                    rdata.obs["anno"] = np.array(ref_cas_anno)
                    rdata = construct_pseudo_bulk(rdata, key="anno", min_samples=self.d_prior)
                sc.tl.pca(rdata, n_comps=self.d_prior)
                cas_prior = torch.FloatTensor(rdata.varm["PCs"].T.copy())
            else:
                sc.tl.pca(sdata, n_comps=self.d_prior)
                cas_prior = torch.FloatTensor(sdata.varm["PCs"].T.copy())
            cas_norm = sdata.X.copy()
            cas_libsize = cas_counts.A.sum(axis=1).reshape(-1, 1)
        else:
            cas_prior = cas_norm = cas_libsize = None
        if self.is_adt:
            sdata = sc.AnnData(adt_counts.copy())
            X_clred, gmean_list = CLR_transform(sdata.X)
            sdata.X = X_clred
            sc.pp.scale(sdata)
            if ref_adt_counts is not None:
                rdata = sc.AnnData(ref_adt_counts.copy())
                X_clred, gmean_list = CLR_transform(rdata.X)
                rdata.X = X_clred
                sc.pp.scale(rdata)
                if ref_adt_anno is not None:
                    rdata.obs["anno"] = np.array(ref_adt_anno)
                    rdata = construct_pseudo_bulk(rdata, key="anno", min_samples=self.d_prior_adt)
                sc.tl.pca(rdata, n_comps=self.d_prior_adt)
                adt_prior = torch.FloatTensor(rdata.varm["PCs"].T.copy())
            else:
                sc.tl.pca(sdata, n_comps=self.d_prior_adt)
                adt_prior = torch.FloatTensor(sdata.varm["PCs"].T.copy())
            adt_norm = sdata.X.copy()
        else:
            adt_prior = adt_norm = None

        self.prior_initialize(rna_prior, cas_prior, adt_prior)
        self.unfreeze()
        self.to(device)
        
        ## First-stage: train basic model
        graph = Integrated_3D_graph(batch_label=batch_label, spatial_mat=spa_mat,
                                    rna_mat=rna_norm, cas_mat=cas_norm, adt_mat=adt_norm,
                                    intra_neighbors=self.intra_neighbors, inter_neighbors=self.inter_neighbors, inter_metric="cosine")
        batch_num_label = StrLabel2Idx(batch_label)
        geo_dataset = GeometricData(graph, node_index=torch.arange(spa_mat.shape[0]))
        transform = torch_geometric.transforms.ToUndirected()
        undirected_geo = transform(geo_dataset)
        cluster_data = ClusterData(undirected_geo, num_parts=int(np.ceil(undirected_geo.num_nodes/batch_size)) * 10, recursive=False, log=False)
        train_loader = ClusterLoader(cluster_data, batch_size=10, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        pbar = tqdm(range(epochs))
        pbar.set_description("First-stage trains basic model")
        early_stopping = EarlyStopping(patience=patience, delta=delta)
        for epoch in pbar:
            self.train()
            train_loss_list = []
            NLL_loss_list = []
            BNN_loss_list = []
            MSE_loss_list = []
            IOA_loss_list = []
            for cur in train_loader:
                np_index = cur.node_index.numpy()
                edge_index = cur.edge_index.to(device)
                edge_attr = cur.edge_attr.to(device)
                batch_indices = data2input(batch_num_label[np_index]).to(device)
                if self.is_rna:
                    rna_counts_batch = data2input(rna_counts[np_index]).to(device)
                    rna_norm_batch = data2input(rna_norm[np_index]).to(device)
                    rna_libsize_batch = data2input(rna_libsize[np_index]).to(device)
                else:
                    rna_counts_batch = rna_norm_batch = rna_libsize_batch = None
                if self.is_cas:
                    cas_counts_batch = data2input(cas_counts[np_index]).to(device)
                    cas_norm_batch = data2input(cas_norm[np_index]).to(device)
                    cas_libsize_batch = data2input(cas_libsize[np_index]).to(device)
                else:
                    cas_counts_batch = cas_norm_batch = cas_libsize_batch = None
                if self.is_adt:
                    adt_norm_batch = data2input(adt_norm[np_index]).to(device)
                else:
                    adt_norm_batch = None
                
                x_lat, nll_loss, mse_loss, ioa_loss, iba_loss, ibp_loss = self.forward(rna_norm_batch, rna_counts_batch, rna_libsize_batch,
                                                                                       cas_norm_batch, cas_counts_batch, cas_libsize_batch,
                                                                                       adt_norm_batch, edge_index,
                                                                                       batch_indices=batch_indices, rna_ridge_lambda=rna_ridge_lambda, cas_ridge_lambda=cas_ridge_lambda, stage=1)
                bnn_loss = self.bnn_loss()
                if self.num_modalities > 1 and self.is_recons:
                    loss = alpha1*bnn_loss + alpha2*ioa_loss + alpha3*nll_loss + alpha4*mse_loss
                elif self.is_recons:
                    loss = alpha1*bnn_loss + alpha3*nll_loss + alpha4*mse_loss
                elif self.num_modalities > 1:
                    loss = alpha1*bnn_loss + alpha2*ioa_loss + alpha3*nll_loss
                else:
                    loss = alpha1*bnn_loss + alpha3*nll_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())
                if not isinstance(nll_loss, int): NLL_loss_list.append(nll_loss.item())
                else: NLL_loss_list.append(nll_loss)
                BNN_loss_list.append(bnn_loss.item())
                if self.is_recons: MSE_loss_list.append(mse_loss.item())
                if self.num_modalities > 1: IOA_loss_list.append(ioa_loss.item())
            
            train_loss = np.mean(train_loss_list)
            early_stopping(train_loss)
            nll_loss = np.mean(NLL_loss_list)
            bnn_loss = np.mean(BNN_loss_list)
            if self.num_modalities > 1 and self.is_recons: 
                ioa_loss = np.mean(IOA_loss_list)
                mse_loss = np.mean(MSE_loss_list)
                pbar.set_postfix({'NLL_loss': nll_loss, 'BNN_loss': bnn_loss, 'MSE_loss':mse_loss, 'IOA_loss': ioa_loss, 'ES counter':early_stopping.counter, "ES patience": early_stopping.patience})
            elif self.is_recons: 
                mse_loss = np.mean(MSE_loss_list)
                pbar.set_postfix({'NLL_loss': nll_loss, 'BNN_loss': bnn_loss, 'MSE_loss':mse_loss, 'ES counter':early_stopping.counter, "ES patience": early_stopping.patience})
            elif self.num_modalities > 1: 
                ioa_loss = np.mean(IOA_loss_list)
                pbar.set_postfix({'NLL_loss': nll_loss, 'BNN_loss': bnn_loss, 'IOA_loss': ioa_loss, 'ES counter':early_stopping.counter, "ES patience": early_stopping.patience})
            else:
                pbar.set_postfix({'NLL_loss': nll_loss, 'BNN_loss': bnn_loss, 'ES counter':early_stopping.counter, "ES patience": early_stopping.patience})
            if early_stopping.early_stop:
                print("Early stop the first-stage training process")
                break
        
        ## Second-stage model training for batch correction
        pbar = tqdm(range(epochs))
        pbar.set_description('Second-stage trains BC model')
        early_stopping = EarlyStopping(patience=patience//2, delta=delta)
        
        disc_optimizer = torch.optim.Adam(self.batch_discriminator.parameters(), lr=lr, weight_decay=weight_decay)
        gen_optimizer = torch.optim.Adam(optim_parameters(self, excluded="batch_discriminator"), 
                                        lr=lr, weight_decay=weight_decay)
        for epoch in pbar:
            joint_lat, _, _ = self.inner_inference(undirected_geo.edge_index, rna_norm, cas_norm, adt_norm, impute=False, batch_fusion=False)
            self.to(device)
            graph = Integrated_3D_graph(batch_label=batch_label, spatial_mat=spa_mat, joint_mat=joint_lat,
                                        rna_mat=None, cas_mat=None, adt_mat=None,
                                        intra_neighbors=self.intra_neighbors, inter_neighbors=self.inter_neighbors)
            geo_dataset = GeometricData(graph, node_index=torch.arange(spa_mat.shape[0]))
            undirected_geo = transform(geo_dataset)
            cluster_data = ClusterData(undirected_geo, num_parts=int(np.ceil(undirected_geo.num_nodes/batch_size)) * 10, recursive=False, log=False)
            train_loader = ClusterLoader(cluster_data, batch_size=10, shuffle=True)

            self.train()
            train_loss_list = []
            NLL_loss_list = []
            MSE_loss_list = []
            BNN_loss_list = []
            IOA_loss_list = []
            IBA_loss_list = []
            IBP_loss_list = []
            DISC_loss_list = []
            GEN_loss_list = []
            for cur in train_loader:
                np_index = cur.node_index.numpy()
                edge_index = cur.edge_index.to(device)
                edge_attr = cur.edge_attr.to(device)
                batch_indices = data2input(batch_num_label[np_index]).to(device)
                batch_indices_np = batch_num_label[np_index]
                joint_lat_batch = data2input(joint_lat[np_index]).to(device)
                positive_indices = []
                negative_indices = []
                for i in range(cur.num_nodes):
                    idx = (edge_index[1]==i) & (edge_attr==-1)
                    num_positive = idx.cpu().numpy().sum()
                    if num_positive > 0: positive_indices.append(edge_index[0, idx])
                    else: positive_indices.append(None)
                    batch_neighbors = edge_index[0, (edge_index[1]==i) & (edge_attr==1)].cpu().numpy()
                    negative_indices_candidate = np.setdiff1d(np.arange(cur.num_nodes)[batch_indices_np==batch_indices_np[i]], batch_neighbors)
                    if negative_samples is None: num_negative = min(num_positive, negative_indices_candidate.shape[0])
                    elif isinstance(negative_samples, int): num_negative = min(num_positive*negative_samples, negative_indices_candidate.shape[0])
                    else: num_negative = int(np.ceil(negative_indices_candidate.shape[0] * negative_samples))
                    selected_indices = np.random.choice(negative_indices_candidate, num_negative, replace=False)
                    negative_indices.append(data2input(selected_indices).to(device))
                if self.is_rna:
                    rna_counts_batch = data2input(rna_counts[np_index]).to(device)
                    rna_norm_batch = data2input(rna_norm[np_index]).to(device)
                    rna_libsize_batch = data2input(rna_libsize[np_index]).to(device)
                else:
                    rna_counts_batch = rna_norm_batch = rna_libsize_batch = None
                if self.is_cas:
                    cas_counts_batch = data2input(cas_counts[np_index]).to(device)
                    cas_norm_batch = data2input(cas_norm[np_index]).to(device)
                    cas_libsize_batch = data2input(cas_libsize[np_index]).to(device)
                else:
                    cas_counts_batch = cas_norm_batch = cas_libsize_batch = None
                if self.is_adt:
                    adt_norm_batch = data2input(adt_norm[np_index]).to(device)
                else:
                    adt_norm_batch = None
                ## update discriminator
                x_lat, rna_lat, cas_lat, adt_lat = self.forward_encoder(rna_norm_batch, cas_norm_batch,
                                                                        adt_norm_batch, edge_index)
                if self.num_modalities == 1: x_lat = self.fusion_layer(x_lat)
                batch_predicted = self.batch_discriminator(x_lat.detach())
                disc_loss = beta3 * F.cross_entropy(batch_predicted, batch_indices)
                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()
                if not isinstance(disc_loss, int): DISC_loss_list.append(disc_loss.item())
                else: DISC_loss_list.append(disc_loss)
                
                ## update generator/encoder and decoder
                x_lat, nll_loss, mse_loss, ioa_loss, iba_loss, ibp_loss = self.forward(rna_norm_batch, rna_counts_batch, rna_libsize_batch,
                                                                                       cas_norm_batch, cas_counts_batch, cas_libsize_batch,
                                                                                       adt_norm_batch, edge_index,
                                                                                       batch_indices, positive_indices, negative_indices, joint_lat_batch,
                                                                                       rna_ridge_lambda=rna_ridge_lambda, cas_ridge_lambda=cas_ridge_lambda, stage=2)
                bnn_loss = self.bnn_loss()
                batch_predicted = self.batch_discriminator(x_lat)
                gen_loss = -F.cross_entropy(batch_predicted, batch_indices)
                
                if self.num_modalities > 1 and self.is_recons:
                    loss = alpha1*bnn_loss + alpha2*ioa_loss + alpha3*nll_loss + alpha4*mse_loss + beta1*iba_loss + beta2*ibp_loss + beta3*gen_loss
                elif self.is_recons:
                    loss = alpha1*bnn_loss + alpha3*nll_loss + alpha4*mse_loss + beta1*iba_loss + beta2*ibp_loss + beta3*gen_loss
                elif self.num_modalities > 1:
                    loss = alpha1*bnn_loss + alpha2*ioa_loss + alpha3*nll_loss + beta1*iba_loss + beta2*ibp_loss + beta3*gen_loss
                else:
                    loss = alpha1*bnn_loss + alpha3*nll_loss + beta1*iba_loss + beta2*ibp_loss + beta3*gen_loss
                gen_optimizer.zero_grad()
                loss.backward()
                gen_optimizer.step()
                
                train_loss_list.append(loss.item() + disc_loss.item())
                if not isinstance(nll_loss, int): NLL_loss_list.append(nll_loss.item())
                else: NLL_loss_list.append(nll_loss)
                if not isinstance(mse_loss, int): MSE_loss_list.append(mse_loss.item())
                else: MSE_loss_list.append(mse_loss)
                if not isinstance(bnn_loss, int): BNN_loss_list.append(bnn_loss.item())
                else: BNN_loss_list.append(bnn_loss)
                if not isinstance(iba_loss, int): IBA_loss_list.append(iba_loss.item()) 
                else: IBA_loss_list.append(iba_loss)
                if not isinstance(iba_loss, int): IBP_loss_list.append(ibp_loss.item())
                else: IBP_loss_list.append(ibp_loss)
                if not isinstance(gen_loss, int): GEN_loss_list.append(gen_loss.item())
                else: GEN_loss_list.append(gen_loss)
            train_loss = np.mean(train_loss_list)
            nll_loss = np.mean(NLL_loss_list)
            bnn_loss = np.mean(BNN_loss_list)
            iba_loss = np.mean(IBA_loss_list)
            ibp_loss = np.mean(IBP_loss_list)
            disc_loss = np.mean(DISC_loss_list)
            gen_loss = np.mean(GEN_loss_list)
            early_stopping(train_loss)
            pbar.set_postfix({'IBA_loss':iba_loss, 'IBP_loss':ibp_loss, 'DISC_loss':disc_loss, 'GEN_loss':gen_loss,
                              'ES counter':early_stopping.counter, "ES patience": early_stopping.patience})
            if early_stopping.early_stop:
                print("Early stop the second-stage training process")
                break
        self.freeze()
        joint_lat, omics_lat, omics_imputed = self.inner_inference(undirected_geo.edge_index, rna_norm, cas_norm, adt_norm, impute=impute, batch_fusion=True)
        
        return joint_lat, omics_lat, omics_imputed
