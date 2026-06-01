# pbmc_obs_export.py
import scanpy as sc

# 讀取 h5ad 檔案，但只需要 metadata
adata = sc.read_h5ad("input_data/pbmcpedia-v20250915-full.h5ad", backed="r")

# 匯出 obs metadata 到 CSV
adata.obs.to_csv("input_data/pbmc_obs.csv")

print("✅ 已成功匯出 pbmc_obs.csv")
