import scanpy as sc

adata = sc.read_h5ad("input_data/pbmcpedia-v20250915-full.h5ad")
adata.var.to_csv("input_data/pbmc_var.csv")
