import numpy as np
import pandas as pd
import scipy
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import completeness_score
import sklearn
import sklearn.neighbors
import scanpy as sc
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import cohen_kappa_score, make_scorer

from .Utils import *

def data_reformat(metrics_df, scenarios, methods=None, samples=None, metrics=None, method_key="model", scenario_key="scenario", sample_key="sample", metrics_key="metrics"):
    index = metrics_df[scenario_key].isin(scenarios)
    if methods is not None: index = index & metrics_df[method_key].isin(methods)
    if samples is not None: index = index & metrics_df[sample_key].isin(samples)
    if metrics is not None: index = index & metrics_df[metrics_key].isin(metrics)
    df = metrics_df.loc[index, :]
    if methods is None: methods = df[method_key].unique()
    samples = df[sample_key].unique()
    
    ncol = len(methods)
    nrow = sum(index) // ncol
    values = np.zeros((nrow, ncol))

    i = 0
    index = []
    for sample in samples:
        for scenario in scenarios:
            metrics = df.loc[(df[sample_key]==sample) & (df[scenario_key]==scenario), metrics_key].unique()
            for metric in metrics:
                for j, method in enumerate(methods):
                    values[i, j] = df.loc[(df[sample_key]==sample) & (df[scenario_key]==scenario) & (df[metrics_key]==metric) & (df[method_key]==method), "score"]
                i += 1
                index.append(sample + "-" + scenario + "-" + metric)
    return pd.DataFrame(values, index=index, columns=methods)

def cal_pval_mat(df, alternative='greater'):
    ndim = df.shape[1]
    results = pd.DataFrame(np.zeros((ndim, ndim)), index=df.columns, columns=df.columns)
    for i in range(ndim):
        for j in range(ndim):
            if i==j: 
                results.iloc[i, j] = np.nan
                continue
            results.iloc[i, j] = scipy.stats.wilcoxon(df.iloc[:, i].values, df.iloc[:, j].values, alternative='greater').pvalue
    return results

def metacell_correlation(raw, imputed, label, metrics="spearman"):
    assert metrics in ("spearman", "pearson"), "metrics should be one of (spearman, pearson)"
        
    label = np.array(label).astype(str)
    result = np.zeros_like(label)
    if scipy.sparse.issparse(raw): raw = raw.A
    if scipy.sparse.issparse(imputed): imputed = imputed.A
    for domain in np.unique(label):
        idx = (label == domain)
        meta = raw[idx, :].mean(axis=0).reshape(1, -1)
        cur = imputed[idx, :]
        if metrics == "pearson": result[idx] = np.corrcoef(meta, cur)[0, 1:]
        elif metrics == "spearman" and cur.shape[0] > 1: result[idx] = scipy.stats.spearmanr(meta, cur, axis=1).correlation[0, 1:]
        else: result[idx] = np.array([scipy.stats.spearmanr(meta, cur, axis=1).correlation])
            
    return result.tolist()

def gene_cell_correlation(expr1, expr2, metrics="spearman"):
    assert metrics in ("spearman", "pearson"), "metrics should be one of (spearman, pearson)"
    assert expr1.shape[0] == expr2.shape[0] and expr1.shape[1] == expr2.shape[1], "shape of expr1 and expr2 should be the same"
    
    cell_corr = []
    gene_corr = []
    if metrics == "pearson":
        for i in range(expr1.shape[0]):
            cell_corr.append(np.corrcoef(expr1[i], expr2[i])[1, 0])
        for j in range(expr1.shape[1]):
            gene_corr.append(np.corrcoef(expr1[:, j], expr2[:, j])[1, 0])
    else:
        for i in range(expr1.shape[0]):
            cell_corr.append(scipy.stats.spearmanr(expr1[i], expr2[i]).correlation)
        for j in range(expr1.shape[1]):
            gene_corr.append(scipy.stats.spearmanr(expr1[:, j], expr2[:, j]).correlation)
            
    return cell_corr, gene_corr

def knn_cross_validation(mtx, label, Kfold=5, k=5, batch_idx=None):
    if not isinstance(label, np.ndarray): label = np.array(label).astype(str)
    target = StrLabel2Idx(label)
    if batch_idx is not None:
        batch_idx = np.array(batch_idx).astype(str)
        groups = StrLabel2Idx(batch_idx)
        split = LeaveOneGroupOut()
        n_jobs = np.unique(batch_idx).shape[0]
    else:
        groups = None
        split = StratifiedKFold(n_splits=Kfold)
        n_jobs = Kfold
    model = KNeighborsClassifier(n_neighbors=k)
    cv_results = cross_validate(model, mtx, target, groups=groups,
                                scoring=("accuracy", "f1_macro", "f1_weighted"),
                                cv=split, n_jobs=n_jobs)
    model = KNeighborsClassifier(n_neighbors=k)
    kappa_score = make_scorer(cohen_kappa_score)
    kappa = cross_validate(model, mtx, target, groups=groups,
                           scoring=kappa_score,
                           cv=split, n_jobs=n_jobs)["test_score"]
    acc, kappa, mf1, wf1 = cv_results["test_accuracy"].mean(), kappa.mean(), cv_results["test_f1_macro"].mean(), cv_results["test_f1_weighted"].mean()
    print('Accuracy: %.3f, Kappa: %.3f, mF1: %.3f, wF1: %.3f' % (acc, kappa, mf1, wf1))
    
    return acc, kappa, mf1, wf1

def cluster_metrics(target, pred):
    target = np.array(target)
    pred = np.array(pred)
    
    ari = adjusted_rand_score(target, pred)
    ami = adjusted_mutual_info_score(target, pred)
    nmi = normalized_mutual_info_score(target, pred)
    fmi = fowlkes_mallows_score(target, pred)
    comp = completeness_score(target, pred)
    homo = homogeneity_score(target, pred)
    print('ARI: %.3f, AMI: %.3f, NMI: %.3f, FMI: %.3f, Comp: %.3f, Homo: %.3f' % (ari, ami, nmi, fmi, comp, homo))
    
    return ari, ami, nmi, fmi, comp, homo

def mean_average_precision(x: np.ndarray, y: np.ndarray, k: int=30, **kwargs) -> float:
    r"""
    Mean average precision
    Parameters
    ----------
    x
        Coordinates
    y
        Cell_type/Layer labels
    k
        k neighbors
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`
    Returns
    -------
    map
        Mean average precision
    """
    
    def _average_precision(match: np.ndarray) -> float:
        if np.any(match):
            cummean = np.cumsum(match) / (np.arange(match.size) + 1)
            return cummean[match].mean().item()
        return 0.0
    
    y = np.array(y)
    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=min(y.shape[0], k + 1), **kwargs).fit(x)
    nni = knn.kneighbors(x, return_distance=False)
    match = np.equal(y[nni[:, 1:]], np.expand_dims(y, 1))
    
    return np.apply_along_axis(_average_precision, 1, match).mean().item()

def rep_metrics(adata, use_rep, key, k_map=30):
    import scib
    if key not in adata.obs or use_rep not in adata.obsm:
        print("KeyError")
        return None
    
    adata.obs[key] = adata.obs[key].astype("category")
    MAP = mean_average_precision(adata.obsm[use_rep].copy(), adata.obs[key], k=k_map)
    cASW = scib.me.silhouette(adata, label_key=key, embed=use_rep)
    cLISI = scib.me.clisi_graph(adata, label_key=key, type_="embed", use_rep=use_rep)
    print('MAP: %.3f, cASW: %.3f, cLISI: %.3f' % (MAP, cASW, cLISI))
    
    return MAP, cASW, cLISI

def batch_metrics(adata, use_rep, batch_key, label_key):
    import scib
    if batch_key not in adata.obs or label_key not in adata.obs or use_rep not in adata.obsm:
        print("KeyError")
        return None
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    adata.obs[label_key] = adata.obs[label_key].astype("category")
    sc.pp.neighbors(adata, use_rep=use_rep)
    GC = scib.me.graph_connectivity(adata, label_key=label_key)
    iLISI = scib.me.ilisi_graph(adata, batch_key=batch_key, type_="embed", use_rep=use_rep)
    kBET = scib.me.kBET(adata, batch_key=batch_key, label_key=label_key, type_="embed", embed=use_rep)
    bASW = scib.me.silhouette_batch(adata, batch_key=batch_key, label_key=label_key, embed=use_rep)
    print('GC: %.3f, iLISI: %.3f, kBET: %.3f, bASW: %.3f' % (GC, iLISI, kBET, bASW))
    
    return GC, iLISI, kBET, bASW

def isolated_metrics(adata, use_rep, label_key, batch_key, threshold=1):
    import scib
    if batch_key not in adata.obs or label_key not in adata.obs or use_rep not in adata.obsm:
        print("KeyError")
        return None
    
    sc.pp.neighbors(adata, use_rep=use_rep)
    isolated_asw = scib.me.isolated_labels_asw(adata, batch_key=batch_key, label_key=label_key, embed=use_rep, iso_threshold=threshold)
    isolated_f1 = scib.me.isolated_labels_f1(adata, batch_key=batch_key, label_key=label_key, embed=use_rep, iso_threshold=threshold)
    print('isolated_asw: %.3f, isolated_f1: %.3f' % (isolated_asw, isolated_f1))
    
    return isolated_asw, isolated_f1