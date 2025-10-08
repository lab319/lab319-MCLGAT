#MCLGAT

The seven scRNA-seq datasets can be downloaded from Gene Expression Omnibus (https://www.ncbi.nlm.nih.gov/geo/) database with the accession numbers GSE75748 (hESC, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75748), GSE81252 (hHEP, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81252), GSE48968 (mDC, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48968), GSE98664 (mESC, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98664) and GSE81682 (mHSC, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81682). All above single-cell datasets with four different kinds of ground-truth networks are available at https://doi.org/10.5281/zenodo.3378975. The lung cancer dataset can be downloaded from Gene Expression Nebulas(https://ngdc.cncb.ac.cn/gen/browse/datasets, GEN ID: GEND000176). The breast cancer dataset can be downloaded from Gene Expression Nebulas(https://ngdc.cncb.ac.cn/gen/browse/datasets, GEN ID: GEND000024).

Requirement
python == 3.7.3
torch == 1.9.1
scikit-learn==1.0.2
numpy==1.19.3
pandas==1.2.4
scipy==1.7.3
Usage
Preparing for gene expression profiles and gene-gene adjacent matrix

MCLGAT integrates gene expression matrix (N×F) with prior gene topology (N×N) to learn low-dimensional vertorized representations with supervision.

Command to run MCLGAT

To train an ab initio model, simply uses the script 'MCLGATmain.py'.

 python MCLGATmain.py
