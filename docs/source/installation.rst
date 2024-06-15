Installation
====

Dependency
----
::

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

These dependencies will be automatically installed along with PRESENT.

Installation via PyPI
----

PRESENT is available on PyPI here_ and can be installed via::

    pip install bio-present
    pip install pyg_lib==0.2.0 torch_sparse==0.6.17 torch_cluster==1.6.1 torch_scatter==2.1.1 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
    

Installation via GitHub
----

PRESENT can also installed from GitHub via::

    git clone https://github.com/lizhen18THU/PRESENT.git
    cd PRESENT
    python setup.py install
    pip install pyg_lib==0.2.0 torch_sparse==0.6.17 torch_cluster==1.6.1 torch_scatter==2.1.1 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

.. _here: https://pypi.org/project/bio-present
