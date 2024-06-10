# Cross-modality representation and multi-sample integration of spatially resolved omics data

## Overview

Spatially resolved sequencing technologies have revolutionized the characterization of biological regulatory processes within microenvironment by simultaneously accessing the states of genomic regions, genes and proteins, along with the spatial coordinates of cells, necessitating advanced computational methods for the cross-modality and multi-sample integrated analysis of spatial omics datasets. To address this gap, we propose PRESENT, an effective and scalable contrastive learning framework, for the cross-modality representation of spatially resolved omics data. Through comprehensive experiments on massive spatially resolved datasets, PRESENT achieves superior performance across various species, tissues, and sequencing technologies, including spatial epigenomics, transcriptomics, and multi-omics. Specifically, PRESENT empowers the incorporation of spatial dependency and complementary omics information simultaneously, facilitating the detection of spatial domains and uncovering biological regulatory mechanisms within microenvironment. Furthermore, PRESENT can be extended to the integrative analysis of horizontal and vertical samples across different dissected regions or developmental stages, thereby promoting the identification of hierarchical structures from a spatiotemporal perspective.

<div align=center>
<img src = "docs/source/PRESENT_Overview.png" width = 100% height = 100%>
</div>



## Installation

### Dependency
```
numpy>=1.24.4
pandas>=2.0.3
scipy>=1.9.3
scikit-learn>=1.3.2
anndata>=0.9.2
networkx>=3.1
scanpy>=1.9.8
episcanpy>=0.3.2
genomicranges>=0.4.2
iranges>=0.2.1
biocutils>=0.1.3
torch>=2.0.0
torch-geometric>=2.3.1
```

### Installation via pypi
PRESENT is available on PyPI [here](https://pypi.org/project/bio-past) and can be installed via
```
pip install bio-present
```

### Installation via Github
You can also install PRESENT from GitHub via
```
git clone https://github.com/lizhen18THU/PRESENT.git
cd PRESENT
python setup.py install
```


## Quick start

### Cross-modality representation of a single sample with PRESENT

### Multi-sample integration with PRESENT-BC


### Find more details and tutorials on [the Documentation of PRESENT](https://past.readthedocs.io/en/latest/).
All the data used in the tutorial are available at [TsinghuaCloudDisk](https://cloud.tsinghua.edu.cn/d/9ab272a99ffb4104a37d/).


## Citation

Zhen Li, Xuejian Cui, Xiaoyang Chen, Zijing Gao, Yuyao Liu, Yan Pan, Shengquan Chen and Rui Jiang. "Cross-modality representation and multi-sample integration of spatially resolved omics data." Preprint at bioRxiv (2023).
