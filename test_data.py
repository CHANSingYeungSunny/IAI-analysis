import os
import pandas as pd
import h5py
import anndata

# 切換到 input_data 資料夾
os.chdir(r"C:\Users\Asus\Desktop\ImmAge+H3N1\input_data")

# 1. PBMC (只讀欄位結構，不展開矩陣)
try:
    # 方法 A: backed 模式
    adata = anndata.read_h5ad("pbmcpedia-v20250915-full.h5ad", backed="r")
    print("PBMC obs columns:", adata.obs.columns.tolist())
    print("PBMC var columns:", adata.var.columns.tolist())
    adata.file.close()
except Exception as e:
    # 方法 B: h5py fallback
    with h5py.File("pbmcpedia-v20250915-full.h5ad", "r") as f:
        print("Top-level groups:", list(f.keys()))
        if "obs" in f:
            print("PBMC obs columns:", list(f["obs"].keys()))
        if "var" in f:
            print("PBMC var columns:", list(f["var"].keys()))

# 2. VDJ
vdj = pd.read_csv("vdj_obs.csv.gz", compression="gzip", nrows=0)
print("VDJ columns:", vdj.columns.tolist())

# 3. GTEx
gtex_expr = pd.read_csv("GTEx_v11_gene_tpm.gct.gz", sep="\t", compression="gzip", nrows=0)
print("GTEx TPM columns:", gtex_expr.columns.tolist())

gtex_meta = pd.read_csv("GTEx_v11_SampleAttributesDS.txt", sep="\t", nrows=0)
print("GTEx Sample Attributes columns:", gtex_meta.columns.tolist())

# 4. WHO FluNet
flu_meta = pd.read_csv("VIW_FLU_METADATA.csv", nrows=0)
print("FluNet Metadata columns:", flu_meta.columns.tolist())

flu_fnt = pd.read_csv("VIW_FNT.csv", nrows=0)
print("FluNet FNT columns:", flu_fnt.columns.tolist())

# 5. CDC VE
cdc_ve = pd.read_csv("CDC_VE_rows.csv", nrows=0)
print("CDC VE columns:", cdc_ve.columns.tolist())
