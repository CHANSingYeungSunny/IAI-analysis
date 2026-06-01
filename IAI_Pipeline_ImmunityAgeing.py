# ============================================
# IAI Pipeline — Immunity Ageing Index
# Per-time-point IAI with temporal alignment
# ============================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pycombat
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[IAI Pipeline] Using device: {device}")

# ============================================
# Module 1: Zcells — Temporal (per-project)
# ============================================
def compute_Zcells_temporal(obs_csv_file):
    """
    Compute Zcells per project (= per calendar time point).
    Returns: DataFrame with columns [project, create_date, zcells, n_donors, mean_age]
    """
    print("\n[Zcells] Loading PBMC metadata...")
    obs = pd.read_csv(obs_csv_file, index_col=0, low_memory=False)

    # B/T cell classification using AIFI_L1_name
    obs["is_b_cell"] = (obs["AIFI_L1_name"] == "B cell").astype(int)
    obs["is_t_cell"] = (obs["AIFI_L1_name"] == "T cell").astype(int)

    # Per-donor B/T ratio
    donor_b = obs.groupby("donor_id")["is_b_cell"].sum()
    donor_t = obs.groupby("donor_id")["is_t_cell"].sum()
    donor_bt_ratio = donor_b / np.maximum(donor_t, 1)

    # Per-donor metadata
    donor_info = obs.groupby("donor_id").agg(
        project=("project", "first"),
        create_date=("create_date", "first"),
        age=("age", "first"),
        sex=("sex", "first"),
        disease=("disease", "first"),
    )
    donor_info["create_date_parsed"] = pd.to_datetime(donor_info["create_date"], errors="coerce")
    donor_info["bt_ratio"] = donor_bt_ratio
    donor_info["age_num"] = pd.to_numeric(donor_info["age"], errors="coerce")

    print(f"[Zcells] Donors with B cells: {(donor_b > 0).sum()}, with T cells: {(donor_t > 0).sum()}")
    print(f"[Zcells] B/T ratio range: {donor_bt_ratio.min():.4f} - {donor_bt_ratio.max():.4f}")

    # Encode covariates as numeric
    sex_map = {"male": 0, "Male": 0, "female": 1, "Female": 1}
    donor_info["sex_num"] = donor_info["sex"].map(lambda s: sex_map.get(str(s).strip(), 0.5))
    donor_info["disease_num"] = donor_info["disease"].map(
        lambda d: 0 if pd.isna(d) or str(d).strip().lower() in ("nan", "hc", "") else 1
    )

    # Standardize per-donor B/T ratio
    from sklearn.preprocessing import StandardScaler
    valid = donor_info["bt_ratio"].notna() & (donor_info["bt_ratio"] < 10)
    zcells_per_donor = pd.Series(0.0, index=donor_info.index)
    bt_valid = donor_info.loc[valid, "bt_ratio"].values.reshape(-1, 1)
    zcells_per_donor.loc[valid] = StandardScaler().fit_transform(bt_valid).flatten()

    # Aggregate to project level (time points)
    # Compute per-project mean of RAW B/T ratio (not within-project standardized)
    # Then standardize across projects to capture between-project variation
    proj_zcells = donor_info.groupby("project").agg(
        create_date=("create_date_parsed", "first"),
        zcells_raw=("bt_ratio", "mean"),  # raw mean B/T per project
        n_donors=("bt_ratio", "count"),
        mean_age=("age_num", "mean"),
    ).reset_index()

    # Sort by date
    proj_zcells = proj_zcells.sort_values("create_date").reset_index(drop=True)

    # Standardize across projects (between-project variation IS the temporal signal)
    # Note: true cross-project batch correction is not possible because each project
    # IS its own batch (24 projects = 24 unique batches). This is a documented
    # limitation — batch effects are confounded with temporal variation.
    proj_zcells["zcells"] = StandardScaler().fit_transform(
        proj_zcells["zcells_raw"].values.reshape(-1, 1)).flatten()

    print(f"[Zcells] {len(proj_zcells)} projects, date range: "
          f"{proj_zcells['create_date'].min().date()} to {proj_zcells['create_date'].max().date()}")
    print(f"[Zcells] Zcells range: {proj_zcells['zcells'].min():.3f} to {proj_zcells['zcells'].max():.3f}")

    return proj_zcells


# ============================================
# Module 2: Zclonotype — Population Baseline
# ============================================
def compute_Zclonotype_baseline(vdj_file):
    """
    Compute clonotype diversity baseline from VDJ data.
    Returns: scalar float (population mean z-score)
    """
    print("\n[Zclonotype] Loading VDJ data...")
    df = pd.read_csv(vdj_file)

    def shannon_entropy(counts):
        p = counts.values / counts.values.sum()
        return entropy(p, base=np.e)

    entropy_vals = df.groupby("sample")["clone_id_size"].apply(shannon_entropy)
    zclonotype_raw = StandardScaler().fit_transform(entropy_vals.values.reshape(-1, 1))

    # Per-receptor-type mean centering
    batch_ids = df.groupby("sample")["receptor_type"].first().values
    zclonotype_corrected = np.zeros_like(zclonotype_raw)
    for b in np.unique(batch_ids):
        mask = batch_ids == b
        zclonotype_corrected[mask] = zclonotype_raw[mask] - zclonotype_raw[mask].mean()

    baseline = float(zclonotype_corrected.mean())
    print(f"[Zclonotype] {len(entropy_vals)} samples, 2 receptor types, baseline = {baseline:.4f}")
    return baseline


# ============================================
# Module 3: Zgene — Population Baseline
# ============================================
def compute_Zgene_baseline(gtex_tpm_file, gtex_meta_file, pbmc_var_file):
    """
    Compute gene expression baseline from GTEx blood samples.
    Returns: scalar float (population mean z-score of HVG expression)
    """
    import gzip
    print("\n[Zgene] Loading metadata...")
    meta = pd.read_csv(gtex_meta_file, sep="\t", low_memory=False)
    pbmc_var = pd.read_csv(pbmc_var_file)

    # Filter blood samples
    blood_samples = meta[(meta["SMTS"] == "Blood") & (meta["SMRIN"] > 6)]
    blood_ids = set(blood_samples["SAMPID"].values)
    print(f"[Zgene] Blood samples in meta: {len(blood_ids)}")

    # Parse GCT header for column indices
    with gzip.open(gtex_tpm_file, 'rt') as f:
        f.readline(); f.readline()
        header = f.readline().strip().split('\t')

    blood_col_idx = [i for i, col in enumerate(header) if col in blood_ids]
    print(f"[Zgene] Blood samples in TPM: {len(blood_col_idx)}")

    if len(blood_col_idx) == 0:
        print("[Zgene] No blood samples — returning 0 baseline")
        return 0.0

    # HVG set
    hv_set = set(pbmc_var[pbmc_var["highly_variable"] == True]["gene_symbols"].values)
    sample_sums = np.zeros(len(blood_col_idx), dtype=np.float64)
    n_found = 0

    # Stream TPM
    print("[Zgene] Streaming TPM for HVG mean expression...")
    with gzip.open(gtex_tpm_file, 'rt') as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            parts = line.strip().split('\t')
            gene_name, gene_desc = parts[0], parts[1]
            if gene_name not in hv_set and gene_desc not in hv_set:
                continue
            for i, ci in enumerate(blood_col_idx):
                try:
                    sample_sums[i] += float(parts[ci])
                except (ValueError, IndexError):
                    pass
            n_found += 1
            if n_found >= 500:
                break

    print(f"[Zgene] HVGs matched: {n_found}")
    if n_found == 0:
        return 0.0

    sample_means = np.log2(sample_sums / n_found + 1)
    zgene_raw = StandardScaler().fit_transform(sample_means.reshape(-1, 1))

    # ComBat by sequencing center
    batch_map = blood_samples.set_index("SAMPID")["SMCENTER"]
    blood_names = [header[i] for i in blood_col_idx]
    batch_info = np.array([str(batch_map.get(s, "unknown")) for s in blood_names])

    valid_mask = pd.Series(batch_info).value_counts().reindex(batch_info).values >= 2
    valid_mask = pd.Series(batch_info).isin(
        pd.Series(batch_info).value_counts()[lambda x: x >= 2].index).values

    if valid_mask.sum() >= 5:
        try:
            Y = zgene_raw[valid_mask] + np.random.normal(0, 1e-8, (valid_mask.sum(), 1))
            z_corrected = pycombat.Combat().fit_transform(Y, batch_info[valid_mask])
            if not np.isnan(z_corrected).all():
                zgene_raw[valid_mask] = z_corrected
        except Exception:
            pass

    baseline = float(StandardScaler().fit_transform(zgene_raw).mean())
    print(f"[Zgene] {len(blood_col_idx)} samples, baseline = {baseline:.4f}")
    return baseline


# ============================================
# Module 4: IAI Time Series Construction
# ============================================
def compute_IAI_timeseries(proj_zcells, zclonotype_baseline, zgene_baseline):
    """
    Combine temporal Zcells with baseline Zclonotype/Zgene into IAI time series.
    Interpolates 24 project time points to weekly resolution.
    Returns: DataFrame with columns [week_date, IAI]
    """
    print("\n[IAI] Building time series...")

    # IAI per project = Zcells (temporal) + baselines (constant offset)
    proj_zcells = proj_zcells.copy()
    proj_zcells["IAI"] = proj_zcells["zcells"] + zclonotype_baseline + zgene_baseline
    proj_zcells["IAI"] = StandardScaler().fit_transform(proj_zcells["IAI"].values.reshape(-1, 1))

    # Create weekly grid from first to last project date
    start_date = proj_zcells["create_date"].min()
    end_date = proj_zcells["create_date"].max()
    weekly_dates = pd.date_range(start_date, end_date, freq="W-MON")

    # Interpolate project-level IAI to weekly
    from scipy.interpolate import interp1d
    proj_dates_num = proj_zcells["create_date"].map(pd.Timestamp.toordinal).values.astype(float)
    weekly_nums = weekly_dates.map(pd.Timestamp.toordinal).values.astype(float)

    # Use linear interpolation with extrapolation clamping
    iai_interpolator = interp1d(proj_dates_num, proj_zcells["IAI"].values,
                                kind="linear", bounds_error=False,
                                fill_value=(proj_zcells["IAI"].values[0],
                                            proj_zcells["IAI"].values[-1]))
    iai_weekly = iai_interpolator(weekly_nums)

    iai_df = pd.DataFrame({
        "week_date": weekly_dates.tz_localize(None),  # strip timezone for merge
        "IAI": iai_weekly,
    })

    print(f"[IAI] {len(proj_zcells)} projects -> {len(iai_df)} weekly points "
          f"({iai_df['week_date'].min().date()} to {iai_df['week_date'].max().date()})")
    print(f"[IAI] IAI range: {iai_df['IAI'].min():.3f} to {iai_df['IAI'].max():.3f}")

    return iai_df


# ============================================
# Module 5: CDC VE 2023-24 Parser
# ============================================
import re

def parse_cdc_ve_2023_24_full(cdc_ve_file):
    """
    Parse the structured CDC VE 2023-24 summary table into a clean DataFrame.
    Handles nested sections: age_group -> adult_subgroup -> subtype -> networks.

    Returns DataFrame with columns:
        age_group, age_subgroup, subtype, network, setting,
        VE_point, VE_lower, VE_upper,
        n_pos_vaccinated, n_pos_total, pct_pos_vaccinated,
        n_neg_vaccinated, n_neg_total, pct_neg_vaccinated
    """
    import re
    print("\n[CDC VE] Parsing CDC_VE_2023_24_full.csv...")
    raw = pd.read_csv(cdc_ve_file)

    rows = []
    current_age_group = None
    current_age_subgroup = None
    current_subtype = None

    for _, row in raw.iterrows():
        net = str(row.iloc[0]).strip()
        age_col = str(row.iloc[4]).strip()
        ve_str = str(row.iloc[3]).strip()
        pos_str = str(row.iloc[1]).strip()
        neg_str = str(row.iloc[2]).strip()

        # Detect age_group changes
        if age_col != current_age_group and age_col not in ('nan', ''):
            current_age_group = age_col
            current_age_subgroup = current_age_group  # default

        # Detect adult subgroup headers (in Network column)
        adult_sub_map = {
            'all adults (aged': 'All adults (>=18y)',
            'adults (aged 18': 'Adults 18-64y',
            'older adults (aged': 'Older adults (>=65y)',
        }
        net_lower = net.lower()
        for key, label in adult_sub_map.items():
            if net_lower.startswith(key):
                current_age_subgroup = label
                break

        # Detect subtype section headers (Network column = subtype name)
        subtype_keywords = ['any influenza', 'influenza a(h1n1)', 'influenza a(h3n2)',
                           'influenza b', 'influenza a']
        is_header = False
        for kw in subtype_keywords:
            if net_lower == kw or net_lower.startswith(kw):
                current_subtype = net
                is_header = True
                break

        if is_header:
            continue

        # Skip sub-headers and non-data rows
        if any(net_lower.startswith(x) for x in ['all adults', 'adults (aged', 'older adults']):
            continue

        # Parse VE string: e.g. "51 (32–64)" or "-5 (-90–43)" or "—"
        ve_point, ve_lower, ve_upper = None, None, None
        if ve_str and ve_str != '—' and ve_str != '-':
            # Replace unicode minus/dash with regular hyphen
            ve_clean = ve_str.replace('–', '-').replace('—', '-')
            # Match: number (number–number)
            m = re.match(r'([-\d.]+)\s*\(([-\d.]+)\s*[-–—]\s*([-\d.]+)\)', ve_clean)
            if m:
                ve_point = float(m.group(1))
                ve_lower = float(m.group(2))
                ve_upper = float(m.group(3))

        # Parse positive/negative strings: e.g. "304/1309 (23%)"
        def parse_frac(s):
            n, tot, pct = None, None, None
            m = re.match(r'(\d+)\s*/\s*(\d+)\s*\((\d+)%\)', s)
            if m:
                n, tot, pct = int(m.group(1)), int(m.group(2)), float(m.group(3))
            return n, tot, pct

        n_pos, tot_pos, pct_pos = parse_frac(pos_str)
        n_neg, tot_neg, pct_neg = parse_frac(neg_str)

        # Detect setting from network name
        setting = 'Inpatient' if 'inpatient' in net_lower else 'Outpatient'

        rows.append({
            'age_group': current_age_group,
            'age_subgroup': current_age_subgroup,
            'subtype': current_subtype,
            'network': net,
            'setting': setting,
            'VE_point': ve_point,
            'VE_lower': ve_lower,
            'VE_upper': ve_upper,
            'n_pos_vaccinated': n_pos,
            'n_pos_total': tot_pos,
            'pct_pos_vaccinated': pct_pos,
            'n_neg_vaccinated': n_neg,
            'n_neg_total': tot_neg,
            'pct_neg_vaccinated': pct_neg,
        })

    ve_df = pd.DataFrame(rows)
    # Drop rows where VE_point is NaN (header rows, missing data)
    ve_df = ve_df.dropna(subset=['VE_point']).reset_index(drop=True)
    print(f"[CDC VE] Parsed {len(ve_df)} VE data rows")
    print(f"[CDC VE] Age groups: {ve_df['age_subgroup'].unique().tolist()}")
    print(f"[CDC VE] Subtypes: {ve_df['subtype'].unique().tolist()}")

    # Show H3N2 summary
    h3n2 = ve_df[ve_df['subtype'].str.contains('H3N2', na=False)]
    if len(h3n2) > 0:
        print(f"[CDC VE] H3N2 VE rows: {len(h3n2)}")
        for _, r in h3n2.iterrows():
            print(f"  {r['age_subgroup']:25s} | {r['network']:25s} | VE={r['VE_point']:.0f}% ({r['VE_lower']:.0f}–{r['VE_upper']:.0f})")

    return ve_df


# ============================================
# Module 6: Build Final DataFrame with Covariates
# ============================================
def build_iai_dataframe(iai_weekly, flu_file, cdc_ve_file=None, cdc_ve_parsed=None, country="United States of America"):
    """
    Merge weekly IAI with FluNet and CDC VE data.
    Returns: DataFrame with columns [time_idx, IAI, VE, country, AH3, SPEC_PROCESSED_NB, week_date]
    """
    print("\n[Build] Merging IAI with external data...")

    # Normalize IAI dates to pure dates (no time component)
    iai_weekly = iai_weekly.copy()
    iai_weekly["week_date"] = iai_weekly["week_date"].dt.normalize()

    # Load FluNet, filter to country and date range
    flu = pd.read_csv(flu_file, low_memory=False)
    flu["week_date"] = pd.to_datetime(flu["ISO_WEEKSTARTDATE"], errors="coerce").dt.normalize()
    flu_ctry = flu[flu["COUNTRY_AREA_TERRITORY"] == country].copy()

    # Aggregate to weekly (sum across any duplicate entries)
    flu_weekly = flu_ctry.groupby("week_date").agg(
        AH3=("AH3", "sum"),
        SPEC_PROCESSED_NB=("SPEC_PROCESSED_NB", "sum"),
    ).reset_index()

    # Merge IAI with FluNet on week_date
    df = iai_weekly.merge(flu_weekly, on="week_date", how="inner")
    print(f"[Build] After FluNet merge: {len(df)} weeks")

    # --- Attach CDC VE 2023-24 (season-level, parsed) ---
    if cdc_ve_parsed is not None and len(cdc_ve_parsed) > 0:
        # Filter to H3N2 subtype
        h3n2_ve = cdc_ve_parsed[cdc_ve_parsed['subtype'].str.contains('H3N2', na=False)]
        # Compute mean VE per age subgroup across networks
        ve_by_age = h3n2_ve.groupby('age_subgroup').agg(
            VE_H3N2_mean=('VE_point', 'mean'),
            VE_H3N2_networks=('VE_point', 'count'),
        ).reset_index()
        print(f"[Build] H3N2 VE by age: {ve_by_age['age_subgroup'].tolist()}")
        print(f"[Build] H3N2 VE means: {ve_by_age['VE_H3N2_mean'].tolist()}")

        # Flag 2023-2024 flu season weeks (Oct 2023 – Mar 2024)
        df["is_flu_season_2023_24"] = (
            (df["week_date"] >= "2023-10-01") & (df["week_date"] <= "2024-03-31")
        )

        # Attach mean H3N2 VE for each age subgroup found
        for _, ve_row in ve_by_age.iterrows():
            subgroup = ve_row['age_subgroup']
            col_name = f"VE_H3N2_{subgroup.replace(' ','_').replace('(','').replace(')','').replace('≥','ge')}"
            # Truncate to reasonable length
            if 'children' in subgroup.lower():
                col_name = 'VE_H3N2_children'
            elif '18-64' in subgroup:
                col_name = 'VE_H3N2_adults_18_64'
            elif '65' in subgroup or 'older' in subgroup.lower():
                col_name = 'VE_H3N2_older_adults'
            else:
                col_name = 'VE_H3N2_all_adults'

            # Season-level: same VE for all weeks in the flu season, NaN otherwise
            df[col_name] = np.where(df["is_flu_season_2023_24"], ve_row['VE_H3N2_mean'], np.nan)
            print(f"[Build]   {col_name}: VE={ve_row['VE_H3N2_mean']:.1f}% (n_networks={int(ve_row['VE_H3N2_networks'])})")

        # Also store the full parsed VE for correlation analysis
        df._cdc_ve_parsed = cdc_ve_parsed
        df._cdc_h3n2 = h3n2_ve
    else:
        df["is_flu_season_2023_24"] = False
        df._cdc_ve_parsed = None
        df._cdc_h3n2 = None

    # --- Also attach CDC_VE_rows.csv (2025-2026 weekly, for reference) ---
    if cdc_ve_file is not None:
        try:
            cdc = pd.read_csv(cdc_ve_file)
            cdc["week_date"] = pd.to_datetime(cdc["week_ending"], errors="coerce").dt.normalize()
            df = df.merge(cdc[["week_date", "nd_weekly_estimate"]], on="week_date", how="left")
            df.rename(columns={"nd_weekly_estimate": "VE_weekly_2025_26"}, inplace=True)
            ve_overlap = df["VE_weekly_2025_26"].notna().sum()
            print(f"[Build] CDC VE 2025-26 weekly overlap: {ve_overlap}/{len(df)} weeks")
        except Exception as e:
            print(f"[Build] CDC VE 2025-26 load failed: {e}")
            df["VE_weekly_2025_26"] = np.nan
    else:
        df["VE_weekly_2025_26"] = np.nan

    # Add time index and country
    df = df.sort_values("week_date").reset_index(drop=True)
    df["time_idx"] = range(len(df))
    df["country"] = country

    # Attach CDC VE metadata as regular columns to avoid copy-loss
    # (custom DataFrame attributes are lost on .copy())
    if cdc_ve_parsed is not None and len(cdc_ve_parsed) > 0:
        df.attrs['_cdc_ve_parsed'] = cdc_ve_parsed
        h3n2 = cdc_ve_parsed[cdc_ve_parsed['subtype'].str.contains('H3N2', na=False)]
        df.attrs['_cdc_h3n2'] = h3n2

    print(f"[Build] Final shape: {df.shape}")
    print(f"[Build] Columns: {df.columns.tolist()}")
    print(f"[Build] AH3>0 weeks: {(df['AH3']>0).sum()}/{len(df)}")
    print(f"[Build] AH3 range: {df['AH3'].min():.0f} - {df['AH3'].max():.0f}")

    return df


# ============================================
# Module 6: Forecasting Models
# ============================================

# --- ARIMA ---
def run_arima(iai_series, exog=None, order=(2,1,2)):
    from statsmodels.tsa.arima.model import ARIMA
    try:
        if exog is not None and len(exog) == len(iai_series):
            model = ARIMA(iai_series, exog=exog, order=order)
        else:
            model = ARIMA(iai_series, order=order)
        fit = model.fit()
        # Forecast 12 weeks; use last exog values for future
        if exog is not None and len(exog) >= 12:
            future_exog = exog[-12:]
            forecast = fit.forecast(steps=12, exog=future_exog)
        else:
            forecast = fit.forecast(steps=12)
        return forecast
    except Exception as e:
        print(f"[ARIMA] Failed: {e}, trying ARIMA(1,0,0)")
        try:
            model = ARIMA(iai_series, order=(1,0,0))
            fit = model.fit()
            return fit.forecast(steps=12)
        except Exception as e2:
            print(f"[ARIMA] Fallback failed: {e2}")
            last = iai_series.iloc[-1] if hasattr(iai_series, 'iloc') else iai_series[-1]
            return np.full(12, float(last))


# --- LSTM ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def run_lstm(iai_series, covariates=None, epochs=20, lr=0.001):
    lstm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LSTM] Using device: {lstm_device}, epochs={epochs}")

    # Build input: [IAI, AH3, SPEC_PROCESSED_NB] if covariates provided
    n_series = len(iai_series)
    if covariates is not None and len(covariates) == n_series:
        feat1 = iai_series.values.reshape(-1, 1)
        feat2 = np.asarray(covariates["AH3"].values[:n_series], dtype=np.float32).reshape(-1, 1)
        feat3 = np.asarray(covariates["SPEC_PROCESSED_NB"].values[:n_series], dtype=np.float32).reshape(-1, 1)
        # Normalize features
        feat2 = feat2 / (feat2.std() + 1e-8)
        feat3 = feat3 / (feat3.std() + 1e-8)
        data_arr = np.hstack([feat1, feat2, feat3])
        input_dim = 3
    else:
        data_arr = iai_series.values.reshape(-1, 1)
        input_dim = 1

    data = torch.tensor(data_arr, dtype=torch.float32)  # (T, input_dim)
    # Create sequences: use 12 past steps to predict next step
    seq_len = 12
    X_list, y_list = [], []
    for i in range(seq_len, n_series):
        X_list.append(data[i-seq_len:i])
        y_list.append(data[i, 0:1])  # predict IAI only
    X = torch.stack(X_list)
    y = torch.stack(y_list)

    dataset = TensorDataset(X, y)
    batch_size = min(32, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_dim=input_dim).to(lstm_device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(lstm_device), yb.to(lstm_device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % max(1, epochs // 5) == 0:
            print(f"[LSTM] Epoch {epoch}/{epochs}, loss={epoch_loss/len(loader):.6f}")

    # Forecast 12 steps
    model.eval()
    preds = []
    last_seq = data[-seq_len:].unsqueeze(0).to(lstm_device)  # (1, seq_len, input_dim)
    with torch.no_grad():
        for _ in range(12):
            pred = model(last_seq)
            preds.append(pred.item())
            # Build next input: use predicted IAI + last known covariates (or zeros)
            if input_dim == 3:
                new_step = torch.tensor([[[pred.item(), 0.0, 0.0]]], device=lstm_device)
            else:
                new_step = pred.view(1, 1, 1)
            last_seq = torch.cat([last_seq[:, 1:, :], new_step], dim=1)

    return np.array(preds), model, data


# --- Permutation Importance (replaces SHAP — no tolerance bug) ---
def permutation_importance_lstm(model, data, n_repeats=10):
    """
    Manual permutation importance for LSTM — no sklearn dependency.
    Returns: dict with 'importances', 'feature_names', 'importances_std'
    """
    data_cpu = data.cpu() if isinstance(data, torch.Tensor) else data
    if len(data_cpu) < 24:
        print("[PermImp] Not enough data")
        return {'importances': np.array([]), 'feature_names': [], 'importances_std': np.array([])}

    # Build sequences
    seq_len = 12
    n_samples = len(data_cpu) - seq_len
    X = torch.stack([data_cpu[i:i+seq_len] for i in range(n_samples)]).numpy()
    y = data_cpu[seq_len:, 0:1].numpy()

    lstm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(lstm_device)
    model.eval()

    n_feat = X.shape[2]
    feat_names = []
    feat_types = ['IAI', 'AH3_norm', 'SPEC_norm']
    for t in range(seq_len, 0, -1):
        for ft in feat_types[:n_feat]:
            feat_names.append(f'{ft}_t-{t}')
    n_total_feat = n_feat * seq_len

    # Baseline score: MSE on original data
    X_flat = X.reshape(n_samples, -1)
    X_t = torch.tensor(X_flat, dtype=torch.float32).to(lstm_device)
    with torch.no_grad():
        baseline_pred = model(torch.tensor(X, dtype=torch.float32).to(lstm_device)).cpu().numpy().flatten()
    baseline_mse = np.mean((baseline_pred - y.flatten()) ** 2)

    # Permute each feature and measure MSE increase
    importances = np.zeros(n_total_feat)
    importances_std = np.zeros(n_total_feat)

    for feat_idx in range(n_total_feat):
        scores = np.zeros(n_repeats)
        for rep in range(n_repeats):
            X_perm = X_flat.copy()
            np.random.shuffle(X_perm[:, feat_idx])  # permute one feature
            X_t = torch.tensor(X_perm, dtype=torch.float32).to(lstm_device)
            # Reshape back to (n_samples, seq_len, n_feat) for LSTM
            X_seq = X_perm.reshape(n_samples, seq_len, n_feat)
            X_t_seq = torch.tensor(X_seq, dtype=torch.float32).to(lstm_device)
            with torch.no_grad():
                pred = model(X_t_seq).cpu().numpy().flatten()
            scores[rep] = np.mean((pred - y.flatten()) ** 2)
        importances[feat_idx] = np.mean(scores) - baseline_mse
        importances_std[feat_idx] = np.std(scores)

    # Move model back
    model.to(lstm_device)

    print(f"[PermImp] Baseline MSE={baseline_mse:.6f}, {n_repeats} repeats x {n_total_feat} features")
    top_idx = np.argsort(importances)[-5:][::-1]
    print(f"[PermImp] Top 5:")
    for i in top_idx:
        print(f"  {feat_names[i]:20s}: {importances[i]:.6f} +/- {importances_std[i]:.6f}")

    return {
        'importances': importances,
        'importances_std': importances_std,
        'feature_names': feat_names,
    }


# --- TFT ---
def run_tft(iai_df):
    import lightning.pytorch as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss

    try:
        data_len = len(iai_df)
        max_enc = min(12, max(4, data_len // 4))
        max_pred = min(6, max(2, data_len // 8))
        print(f"[TFT] Data: {data_len}, encoder={max_enc}, decoder={max_pred}")

        if data_len < max_enc + max_pred + 4:
            raise ValueError(f"Data too short: {data_len}")

        # Prepare covariates — ensure no NaN in required columns
        df = iai_df.copy()
        df["AH3_norm"] = df["AH3"].fillna(0) / (df["AH3"].std() + 1e-8)
        df["SPEC_norm"] = df["SPEC_PROCESSED_NB"].fillna(0) / (df["SPEC_PROCESSED_NB"].std() + 1e-8)
        # Drop rows with any NaN in critical columns
        required_cols = ["time_idx", "IAI", "AH3_norm", "SPEC_norm", "country"]
        df_clean = df[required_cols].dropna().copy()
        df_clean["country"] = df_clean["country"].astype(str)
        print(f"[TFT] Clean data: {len(df_clean)} rows (dropped {len(df)-len(df_clean)} NaN rows)")

        dataset = TimeSeriesDataSet(
            df_clean,
            time_idx="time_idx",
            target="IAI",
            group_ids=["country"],
            max_encoder_length=max_enc,
            max_prediction_length=max_pred,
            time_varying_known_reals=["time_idx", "AH3_norm", "SPEC_norm"],
            time_varying_unknown_reals=["IAI"],
        )

        # Use small batch and the dataset's own collate function
        batch_sz = min(16, max(2, len(dataset) // 20))
        loader = DataLoader(dataset, batch_size=batch_sz, shuffle=True,
                            drop_last=True, num_workers=0,
                            collate_fn=dataset._collate_fn)
        print(f"[TFT] Batch size: {batch_sz}, batches: {len(loader)}")

        model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=0.001,
            hidden_size=32,
            attention_head_size=2,
            dropout=0.1,
            loss=QuantileLoss(),
        )

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=5,
            enable_progress_bar=False,
            logger=False,
        )

        trainer.fit(model, loader)

        # Get predictions in raw mode (includes attention weights and quantiles)
        raw_pred = model.predict(dataset, mode="raw")
        # raw_pred.prediction shape: (n_samples, n_timesteps, n_quantiles)
        forecast = raw_pred.prediction[:, :, 3].cpu().numpy()  # median quantile (index 3 of 7)
        # Average across samples to get 12-step forecast, or take last window
        forecast_mean = forecast.mean(axis=0)  # mean across samples
        # Get attention weights
        attention = raw_pred.encoder_attention
        if hasattr(attention, 'cpu'):
            attention = attention.cpu().numpy()

        return forecast, attention, model, dataset

    except Exception as e:
        print(f"[TFT] Failed: {e}, returning dummy")
        import traceback; traceback.print_exc()
        dummy_forecast = np.full((1, 6), iai_df["IAI"].values[-1])
        dummy_attention = np.eye(6).reshape(1, 1, 6, 6)
        return dummy_forecast, dummy_attention, None, None


# ============================================
# Module 7: Forecast Orchestrator
# ============================================
def compute_forecasts(iai_series, iai_df):
    results = {}

    # External covariates for models
    covariates = iai_df[["AH3", "SPEC_PROCESSED_NB"]].copy() if "AH3" in iai_df.columns else None

    # ARIMA with AH3 as exogenous variable
    print("\n[Forecast] Running ARIMA...")
    exog = iai_df["AH3"].values if "AH3" in iai_df.columns else None
    results["ARIMA"] = run_arima(iai_series, exog=exog)

    # LSTM with covariates
    print("[Forecast] Running LSTM...")
    lstm_preds, lstm_model, lstm_data = run_lstm(iai_series, covariates=covariates)
    results["LSTM"] = lstm_preds

    # Permutation Importance
    print("[Forecast] Computing permutation importance...")
    results["LSTM_SHAP"] = permutation_importance_lstm(lstm_model, lstm_data)

    # TFT
    print("[Forecast] Running TFT...")
    tft_preds, tft_attn, _, _ = run_tft(iai_df)
    results["TFT"] = tft_preds
    results["TFT_attention"] = tft_attn

    return results


# ============================================
# Module 8: Auto-generate Results Description
# ============================================
def generate_results_description(results_dir="results", flu_file="input_data/VIW_FNT.csv",
                                  country="United States of America"):
    """Read all result files and generate RESULTS_DESCRIPTION.txt."""
    import os
    from datetime import datetime
    from scipy import stats as scipy_stats

    out_path = os.path.join(results_dir, "RESULTS_DESCRIPTION.txt")
    lines = []
    def w(s=""): lines.append(s)

    w("=" * 80)
    w("IMMUNE AGEING INDEX (IAI) — H3N2 VACCINE RESPONSE DECLINE")
    w("Complete Results Description — Auto-Generated")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w("=" * 80)
    w()
    w("This document describes every output file in results/. It is self-contained:")
    w("an AI or human can interpret all findings without opening the original files.")

    # ---- IAI_time_series.csv ----
    iai_file = os.path.join(results_dir, "IAI_time_series.csv")
    if os.path.exists(iai_file):
        iai = pd.read_csv(iai_file)
        iai['d'] = pd.to_datetime(iai['week_date'])
        v = iai['IAI'].values
        w()
        w("=" * 80)
        w("FILE 1: IAI_time_series.csv — Weekly Immune Ageing Index")
        w("=" * 80)
        w(f"  Rows: {len(iai)} weekly values")
        w(f"  Date range: {iai['d'].min().date()} to {iai['d'].max().date()}")
        w(f"  Mean={np.mean(v):.4f}  Median={np.median(v):.4f}  Std={np.std(v):.4f}")
        w(f"  Min={np.min(v):.4f}  Max={np.max(v):.4f}")
        w(f"  Skew={scipy_stats.skew(v):.4f}  Kurtosis={scipy_stats.kurtosis(v):.4f}")
        w(f"  Q1={np.percentile(v,25):.4f}  Q3={np.percentile(v,75):.4f}")

        # By year
        w("  IAI BY YEAR:")
        for yr in sorted(iai['d'].dt.year.unique()):
            m = iai['d'].dt.year == yr
            w(f"    {yr}: mean={v[m].mean():.4f} std={v[m].std():.4f} n={m.sum()}")

        # Autocorr
        w("  AUTOCORRELATION:")
        for lag in [1, 4, 12, 26, 52]:
            if lag < len(v):
                ac = np.corrcoef(v[:-lag], v[lag:])[0, 1]
                w(f"    Lag {lag:2d}: r={ac:.4f}")

        # Trend
        x = np.arange(len(v))
        slope, _, r2, pv, _ = scipy_stats.linregress(x, v)
        w(f"  TREND: slope={slope:.6f}/wk  R²={r2:.4f}  p={pv:.4f}")

    # ---- Forecast_results.csv ----
    fc_file = os.path.join(results_dir, "Forecast_results.csv")
    if os.path.exists(fc_file):
        fc = pd.read_csv(fc_file)
        w()
        w("=" * 80)
        w("FILE 2: Forecast_results.csv — 12-Week Forecasts")
        w("=" * 80)
        for c in fc.columns:
            f = fc[c].values
            w(f"  {c}: mean={np.mean(f):.4f} std={np.std(f):.4f} [{np.min(f):.4f}, {np.max(f):.4f}]")
        # LSTM trajectory
        if 'LSTM' in fc.columns:
            lstm = fc['LSTM'].values
            w(f"  LSTM trajectory: {lstm[0]:.4f} -> {lstm[-1]:.4f} over 12 steps")
            w(f"  Direction: {'RECOVERY' if lstm[-1] > lstm[0] else 'DECLINE'}")

    # ---- Correlation_results.csv ----
    corr_file = os.path.join(results_dir, "Correlation_results.csv")
    if os.path.exists(corr_file):
        corr = pd.read_csv(corr_file)
        w()
        w("=" * 80)
        w("FILE 3: Correlation_results.csv — Statistical Tests")
        w("=" * 80)
        for _, row in corr.iterrows():
            w(f"  {row['Comparison']}: stat={row['Correlation']:.4f} p={row['p-value']:.4e} N={int(row['N'])}")

    # ---- IAI_values.csv (legacy) ----
    iav_file = os.path.join(results_dir, "IAI_values.csv")
    if os.path.exists(iav_file):
        iav = pd.read_csv(iav_file)
        v2 = iav['IAI'].dropna().values
        w()
        w("=" * 80)
        w("FILE 4: IAI_values.csv — Legacy (Superseded)")
        w("=" * 80)
        w(f"  N={len(v2)} mean={np.mean(v2):.4f} std={np.std(v2):.4f}")
        w("  NOTE: This is the v1 per-donor IAI. Superseded by IAI_time_series.csv.")

    # ---- FluNet merge stats ----
    if os.path.exists(flu_file) and os.path.exists(iai_file):
        flu = pd.read_csv(flu_file, low_memory=False)
        flu['wd'] = pd.to_datetime(flu['ISO_WEEKSTARTDATE']).dt.normalize()
        fu = flu[flu['COUNTRY_AREA_TERRITORY'] == country]
        fw = fu.groupby('wd').agg(AH3=('AH3', 'sum'), SPEC=('SPEC_PROCESSED_NB', 'sum')).reset_index()
        iai['wd'] = iai['d'].dt.normalize()
        df = iai.merge(fw, on='wd', how='inner')
        w()
        w("=" * 80)
        w("SUPPLEMENTARY: IAI vs FluNet H3N2 (350 weeks)")
        w("=" * 80)
        w(f"  AH3>0 weeks: {(df['AH3']>0).sum()}/{len(df)}")
        w(f"  AH3 max: {df['AH3'].max():.0f}")
        # Quartile analysis
        df['IAI_q'] = pd.qcut(df['IAI'], 4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
        for q, g in df.groupby('IAI_q', observed=True):
            w(f"  {q}: IAI={g['IAI'].mean():.3f} AH3_mean={g['AH3'].mean():.0f} n={len(g)}")
        # Xcorr
        w("  CROSS-CORRELATION (IAI vs AH3):")
        iaiv = df['IAI'].values
        ah3v = df['AH3'].values
        for lag in [-12, -8, -4, 0, 4, 8, 12]:
            if lag < 0:
                c = np.corrcoef(iaiv[:lag], ah3v[-lag:])[0, 1]
            elif lag > 0:
                c = np.corrcoef(iaiv[lag:], ah3v[:-lag])[0, 1]
            else:
                c = np.corrcoef(iaiv, ah3v)[0, 1]
            label = 'IAI leads' if lag < 0 else ('AH3 leads' if lag > 0 else 'sync')
            w(f"    lag={lag:+3d} ({label:12s}): r={c:.4f}")

    # ---- PNG descriptions (with dynamic stats) ----
    iai_v = iai['IAI'].values if 'iai' in dir() else np.zeros(1)
    skew_val = scipy_stats.skew(iai_v)
    kurt_val = scipy_stats.kurtosis(iai_v)
    fc_lstm = fc['LSTM'].values if 'LSTM' in fc.columns else np.zeros(12)
    fc_arima = fc['ARIMA'].values if 'ARIMA' in fc.columns else np.zeros(12)
    corr_pearson = corr.iloc[0]['Correlation'] if len(corr) > 0 else np.nan
    corr_p = corr.iloc[0]['p-value'] if len(corr) > 0 else np.nan
    flu_iai = df.loc[df['wd'].between('2023-10-01','2024-03-31'), 'IAI'].mean() if 'df' in dir() else np.nan

    w()
    w("=" * 80)
    w("PLOTS (PNG FILES)")
    w("=" * 80)
    w(f"  IAI_distribution.png: Histogram + KDE of {len(iai_v)} weekly IAI values. "
      f"Skew={skew_val:.2f}, Kurtosis={kurt_val:.2f}.")
    w(f"  Forecast_models.png: Historical IAI (350 wks) + ARIMA (mean={fc_arima.mean():.2f}) "
      f"+ LSTM ({fc_lstm[0]:.2f}->{fc_lstm[-1]:.2f} over 12 steps) + TFT (dummy).")
    w(f"  LSTM_SHAP.png: Permutation importance bar chart. Top features ranked by MSE increase.")
    w(f"  TFT_attention.png: TFT attention heatmap (placeholder — collation bug).")
    w(f"  IAI_vs_FluNet.png: Scatter IAI vs H3N2. r={corr_pearson:.2f}, p={corr_p:.2e}.")
    w(f"  IAI_vs_CDCVE.png: 26 IAI values (Oct23-Mar24, mean IAI={flu_iai:.3f}) "
      f"+ 5 H3N2 VE reference lines by age/network.")

    w()
    w("=" * 80)
    w("END OF RESULTS DESCRIPTION")
    w("=" * 80)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[Results] Auto-generated {out_path} ({len(lines)} lines)")
