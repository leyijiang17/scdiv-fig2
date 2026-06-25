# scdiv-fig2
Experimental code related to Figure 2 of the manuscript 'Diversity in Transcriptomics without Cell Types' 

### Dataset
The mouse development single-cell atlas is available on the NCBI Gene Expression Omnibus (GEO) under accession number [GSE228590](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE228590). 

### Dataset overview
The mouse development dataset consists of 11.4 million nuclei derived from 74 embryos, sampled at 6-hour intervals across 43 time points spanning late gastrulation (E8) to birth (postnatal day (P)0). 

### Dataset reference
C. Qiu, B. K. Martin, I. C. Welsh, R. M. Daza, T.-M. Le, X. Huang, E. K. Nichols, M. L. Taylor, O. Fulton, D. R. O’Day et al. A single-cell time-lapse of mouse prenatal development from gastrula to birth. Nature, 626:1084–1093, Feb. 2024. https://doi.org/10.1038/s41586-024-07069-w.

### Data preprocessing
Following the quality control standards outlined in the paper, we identified the top 2,000 highly variable genes (HVGs) per embryonic stage using Seurat's `FindVariableFeatures`. To ensure robustness across developmental stages, we retained only the genes flagged as variable in more than three independent time points, resulting in a final set of 3,754 HVGs. All separate run data were then integrated into a single, unified `.h5ad` file restricted to these features, and cell type annotations from the paper's supplementary table were appended directly to the metadata.
