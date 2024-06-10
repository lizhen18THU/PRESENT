from setuptools import setup, find_packages

setup(
    name = "bio-present",
    version = "0.0.4",
    keywords = ["pip", "present", "spatial omics"],
    description = "Cross-modality representation and multi-sample integration of spatially resolved omics data",
    long_description = "Spatially resolved sequencing technologies have revolutionized the characterization of biological regulatory processes within microenvironment by simultaneously accessing the states of genomic regions, genes and proteins, along with the spatial coordinates of cells, necessitating advanced computational methods for the cross-modality and multi-sample integrated analysis of spatial omics datasets. To address this gap, we propose PRESENT, an effective and scalable contrastive learning framework, for the cross-modality representation of spatially resolved omics data. Through comprehensive experiments on massive spatially resolved datasets, PRESENT achieves superior performance across various species, tissues, and sequencing technologies, including spatial epigenomics, transcriptomics, and multi-omics. Specifically, PRESENT empowers the incorporation of spatial dependency and complementary omics information simultaneously, facilitating the detection of spatial domains and uncovering biological regulatory mechanisms within microenvironment. Furthermore, PRESENT can be extended to the integrative analysis of horizontal and vertical samples across different dissected regions or developmental stages, thereby promoting the identification of hierarchical structures from a spatiotemporal perspective.",
    license = "MIT License",
    url = "https://github.com/lizhen18THU/PRESENT",
    author = "Zhen Li",
    author_email = "lizhen18@tsinghua.org.cn",
    packages = find_packages(),
    python_requires = ">3.6.0",
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    install_requires=[
        'numpy>=1.24.4',
        'pandas>=2.0.3',
        'scipy>=1.9.3',
        'scikit-learn>=1.3.2',
        'anndata>=0.9.2',
        'networkx>=3.1',
        'scanpy>=1.9.8',
        'episcanpy==0.3.2',
        'genomicranges>=0.4.2',
        'iranges>=0.2.1',
        'biocutils>=0.1.3',
        'torch>=2.0.0',
        'torch-geometric>=2.3.1',
    ]
)
