"""
make_dataset_v2.py -- two sources, one dataset: EnerNOC (5-min, 2012) + Building Data Genome 2 (hourly, 2016-17)
-> harmonised 'bills -> load-shape archetype' table.  Writes to data/processed_v2/ (v1 untouched).
Label is defined on HOURLY normalised weekday/weekend profiles for every source, so it does not depend on resolution.
Run from the repository root: python3 src/make_dataset_v2.py   (EnerNOC ~25 s + BDG2 ~2 min on a laptop)
"""
from pathlib import Path
import json, warnings, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent;  # repository root (this file is in src/)
RAW_E = ROOT / "data/raw/EnerNOC csv-only"; RAW_B = ROOT / "data/raw/bdg2"
OUT = ROOT / "data/processed_v2"; OUT.mkdir(parents=True, exist_ok=True); FIG = ROOT / "figures_v2"; FIG.mkdir(exist_ok=True)
SEED = 42; OFFSETS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

# ---------------------------------------------------------------- harmonised building types
BT_ENERNOC = {"Commercial Property": "office_commercial", "Education": "education", "Food Sales & Storage": "retail_food",
              "Light Industrial": "industrial"}
BT_BDG2 = {"Education": "education", "Office": "office_commercial", "Entertainment/public assembly": "assembly_public",
           "Public services": "assembly_public", "Lodging/residential": "lodging", "Healthcare": "healthcare",
           "Warehouse/storage": "industrial", "Manufacturing/industrial": "industrial", "Retail": "retail_food",
           "Food sales and service": "retail_food", "Services": "office_commercial", "Technology/science": "office_commercial",
           "Utility": "other", "Parking": "other", "Other": "other", "Religious worship": "assembly_public"}
def region_from_tz(tz):
    if tz.startswith("Europe/London"): return "UK"
    if tz.startswith("Europe/Dublin"): return "Ireland"
    return {"Eastern": "US_East", "New_York": "US_East", "Central": "US_Central", "Chicago": "US_Central",
            "Mountain": "US_Mountain", "Denver": "US_Mountain", "Phoenix": "US_Mountain", "Pacific": "US_West", "Los_Angeles": "US_West"}.get(tz.split("/")[-1], "other")

# ---------------------------------------------------------------- common feature extraction on an HOURLY local-time series (kW)
def bills_from_hourly(h, year, offset_days=0, peak_sub=None):
    """Monthly bills from an hourly kW series. peak_sub: optional finer series (15-min) for the billed peak."""
    idx = h.index - pd.Timedelta(days=offset_days)
    idx = idx.where(idx.year == year, idx + pd.Timedelta(days=366 if pd.Timestamp(year, 12, 31).dayofyear == 366 else 365))
    rows = []
    for (y, m), v in h.groupby([idx.year, idx.month]):
        if y != year or v.isna().mean() > 0.2 or len(v) < 600: continue
        pk15 = np.nan
        if peak_sub is not None:
            i2 = peak_sub.index - pd.Timedelta(days=offset_days); i2 = i2.where(i2.year == year, i2 + pd.Timedelta(days=366))
            vv = peak_sub[(i2.year == y) & (i2.month == m)]; pk15 = vv.max()
        rows.append(dict(month=m, kwh=v.sum(), peak_kw_hourly=v.max(), peak_kw_15min=pk15, load_factor=v.mean() / v.max()))
    return pd.DataFrame(rows)

def shape_from_hourly(h):
    v = h.dropna(); peak = v.max(); mean = v.mean()
    wd = v[v.index.dayofweek < 5]; we = v[v.index.dayofweek >= 5]
    pwd = wd.groupby(wd.index.hour).mean().reindex(range(24)).interpolate(); pwe = we.groupby(we.index.hour).mean().reindex(range(24)).interpolate()
    pwd_n = pwd / pwd.mean(); pwe_n = pwe / pwd.mean()
    day = pwd_n.loc[9:16].mean(); night = pwd_n.loc[0:5].mean(); eve = pwd_n.loc[17:21].mean()
    win = v[(v.index.month.isin([12, 1, 2, 3])) & (v.index.dayofweek < 5) & (v.index.hour >= 16) & (v.index.hour < 20)]
    return dict(annual_kwh=v.sum(), peak_kw_hourly=peak, load_factor=mean / peak,
                hours_above_70=float((v > .7 * peak).sum()), hours_above_80=float((v > .8 * peak).sum()), hours_above_90=float((v > .9 * peak).sum()),
                nb_window_coincidence=win.max() / peak if len(win) else np.nan,
                day_night_ratio=day / night, evening_day_ratio=eve / day, weekend_weekday_ratio=pwe_n.mean() / pwd_n.mean(),
                peak_hour_wd=int(pwd_n.idxmax())), pwd_n.values, pwe_n.values

records, PWD, PWE, bills, bills_aug = [], [], [], [], []
def add(uid, h, year, meta, peak_sub=None, extra=None):
    b = bills_from_hourly(h, year, 0, peak_sub)
    if len(b) < 12: return False
    tgt, pwd, pwe = shape_from_hourly(h)
    if extra: tgt.update(extra)
    rec = dict(uid=uid, **meta, year=year, **tgt); records.append(rec); PWD.append(pwd); PWE.append(pwe)
    b.insert(0, "uid", uid); bills.append(b)
    for off in OFFSETS:
        ba = bills_from_hourly(h, year, off, peak_sub)
        if len(ba) == 12: ba.insert(0, "uid", uid); ba.insert(1, "offset_days", off); bills_aug.append(ba)
    return True

# ---------------------------------------------------------------- EnerNOC
meta_e = pd.read_csv(RAW_E / "meta/all_sites.csv").set_index("SITE_ID")
for sid in meta_e.index:
    df = pd.read_csv(RAW_E / f"csv/{sid}.csv")
    t = pd.to_datetime(df.dttm_utc, utc=True).dt.tz_convert(meta_e.loc[sid, "TIME_ZONE"]).dt.tz_localize(None)
    s = pd.Series(df.value.values * 12.0, index=t); s = s[~s.index.duplicated()]; s = s[s > 0].asfreq("5min")
    gap = int(s.isna().sum()); s = s.interpolate(limit=12, limit_direction="both")
    s15 = s.resample("15min").mean(); h = s.resample("1h").mean(); h = h[h.index.year == 2012]; s15 = s15[s15.index.year == 2012]
    if gap / 105408 >= 0.10: continue
    m = dict(source="enernoc", source_id=str(sid), building_type=BT_ENERNOC[meta_e.loc[sid, "INDUSTRY"]], sub_type=meta_e.loc[sid, "SUB_INDUSTRY"],
             sq_ft=float(meta_e.loc[sid, "SQ_FT"]), lat=meta_e.loc[sid, "LAT"], lng=meta_e.loc[sid, "LNG"], tz=meta_e.loc[sid, "TIME_ZONE"],
             region=region_from_tz(meta_e.loc[sid, "TIME_ZONE"]), resolution_min=15, pct_missing=gap / 105408 * 100)
    add(f"enernoc_{sid}", h, 2012, m, peak_sub=s15, extra=dict(peak_kw_15min=float(s15.max()), interval_uplift=float(s15.max() / h.max())))
print("EnerNOC done:", len(records))

# ---------------------------------------------------------------- BDG2 (hourly kWh per hour == average kW; timestamps are site-local)
meta_b = pd.read_csv(RAW_B / "metadata.csv").set_index("building_id")
E = pd.read_csv(RAW_B / "electricity_cleaned.csv", index_col=0, parse_dates=True, dtype="float32")
n_b = 0
for bid in E.columns:
    if bid not in meta_b.index: continue
    mb = meta_b.loc[bid]
    if pd.isna(mb.sqft) or pd.isna(mb.timezone): continue
    col = E[bid]
    for year in (2016, 2017):
        h = col[col.index.year == year].astype(float); h = h.where(h > 0)
        miss = h.isna().mean()
        if miss >= 0.10: continue
        h = h.interpolate(limit=3, limit_direction="both")
        m = dict(source="bdg2", source_id=bid, building_type=BT_BDG2.get(mb.primaryspaceusage, "other"), sub_type=mb.sub_primaryspaceusage,
                 sq_ft=float(mb.sqft), lat=mb.lat, lng=mb.lng, tz=mb.timezone, region=region_from_tz(mb.timezone), resolution_min=60, pct_missing=miss * 100)
        if add(f"bdg2_{bid}_{year}", h, year, m, extra=dict(peak_kw_15min=np.nan, interval_uplift=np.nan)): n_b += 1
print("BDG2 building-years:", n_b)

sites = pd.DataFrame(records).set_index("uid"); PWD = np.vstack(PWD); PWE = np.vstack(PWE)
bills = pd.concat(bills, ignore_index=True); bills_aug = pd.concat(bills_aug, ignore_index=True)

# ---------------------------------------------------------------- duplicate detection (within source), group key = building
X = np.nan_to_num(np.hstack([PWD, PWE]), nan=1.0)
sites["building_key"] = np.where(sites.source == "bdg2", "bdg2_" + sites.source_id.astype(str), "enernoc_" + sites.source_id.astype(str))
sites["dup_of"] = ""
for src in ("enernoc", "bdg2"):
    idx = np.where(sites.source.values == src)[0]
    C = np.corrcoef(X[idx]); keys = sites.building_key.values[idx]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if keys[a] != keys[b] and C[a, b] > 0.9995 and abs(np.log(sites.annual_kwh.iloc[idx[a]] / sites.annual_kwh.iloc[idx[b]])) < 3:
                # same shape, different building id -> treat as duplicate meter; keep the first
                if sites.dup_of.iloc[idx[b]] == "": sites.iloc[idx[b], sites.columns.get_loc("dup_of")] = sites.index[idx[a]]
# EnerNOC: known copies also have identical gap patterns; the shape test above catches them
sites["keep"] = sites.dup_of == ""
# conflicting building type across EnerNOC copies -> unknown
grp = sites[sites.source == "enernoc"].copy(); grp["g"] = np.where(grp.dup_of == "", grp.index, grp.dup_of)
conflict = grp.groupby("g").building_type.nunique() > 1
sites.loc[conflict[conflict].index, "building_type"] = "unknown"
keep = sites.keep.values; sites_all = sites.copy(); sites = sites[keep].copy(); PWD, PWE, X = PWD[keep], PWE[keep], X[keep]
print("kept rows:", len(sites), sites.source.value_counts().to_dict(), "| unique buildings:", sites.building_key.nunique())

# ---------------------------------------------------------------- sanity filter: degenerate meters (spike-dominated)
ok = (sites.load_factor > 0.05) & np.isfinite(sites.day_night_ratio) & np.isfinite(sites.weekend_weekday_ratio)
print("dropped degenerate meters:", int((~ok).sum())); sites = sites[ok].copy(); PWD, PWE, X = PWD[ok.values], PWE[ok.values], X[ok.values]

# ---------------------------------------------------------------- archetype labels: explicit rules on two measured shape descriptors
# (thresholds are stated hyperparameters, see data/README.md). k-means is run only as a check.
DN_FLAT, DN_DAY, WE_7DAY = 1.3, 1.8, 0.85
def rule_label(dn, we):
    if dn < DN_FLAT: return "flat_7day"
    if we >= WE_7DAY: return "daytime_7day"
    if dn >= DN_DAY: return "daytime_weekday"
    return "weekday_moderate"
sites["archetype"] = [rule_label(a, b) for a, b in zip(sites.day_night_ratio, sites.weekend_weekday_ratio)]
# distance to the nearest decision boundary, in units of the descriptor -> small = borderline ("soft") label
sites["label_margin"] = np.minimum.reduce([abs(sites.day_night_ratio - DN_FLAT), abs(sites.day_night_ratio - DN_DAY), abs(sites.weekend_weekday_ratio - WE_7DAY)]).round(3)
classes = ["flat_7day", "daytime_7day", "daytime_weekday", "weekday_moderate"]; sites["cluster"] = sites.archetype.map({c: i for i, c in enumerate(classes)})
K = 4; names = dict(enumerate(classes))
# k-means check: how well does an unsupervised K=4 partition agree with the rules?
from sklearn.metrics import adjusted_rand_score
km = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(X); ari = adjusted_rand_score(sites.cluster, km.labels_)
sil = {4: round(float(silhouette_score(X, km.labels_, sample_size=2000, random_state=SEED)), 3)}
print("k-means(4) vs rule labels: adjusted Rand =", round(ari, 3))

# ---------------------------------------------------------------- ML tables (billed peak = 15-min where available else hourly, with a flag)
bills["peak_kw_billed"] = bills.peak_kw_15min.fillna(bills.peak_kw_hourly); bills_aug["peak_kw_billed"] = bills_aug.peak_kw_15min.fillna(bills_aug.peak_kw_hourly)
META = ["source", "building_key", "building_type", "sub_type", "sq_ft", "lat", "lng", "region", "resolution_min", "year", "archetype", "cluster", "label_margin", "hours_above_80"]
def wide(b, keys):
    w = b.pivot_table(index=keys, columns="month", values=["kwh", "peak_kw_billed"]); w.columns = [f"{a}_m{m:02d}" for a, m in w.columns]; return w.reset_index()
X_wide = wide(bills[bills.uid.isin(sites.index)], ["uid"]).merge(sites[META], left_on="uid", right_index=True).dropna()
X_aug = wide(bills_aug[bills_aug.uid.isin(sites.index)], ["uid", "offset_days"]).merge(sites[META], left_on="uid", right_index=True).dropna()
sites.to_csv(OUT / "sites.csv"); sites_all.to_csv(OUT / "sites_all_with_duplicates.csv"); bills.to_csv(OUT / "bills.csv", index=False)
X_wide.to_csv(OUT / "dataset.csv", index=False); X_aug.to_csv(OUT / "dataset_aug.csv", index=False)
pd.DataFrame(PWD, index=sites.index, columns=[f"h{i:02d}" for i in range(24)]).to_csv(OUT / "profiles_weekday_hourly.csv")
pd.DataFrame(PWE, index=sites.index, columns=[f"h{i:02d}" for i in range(24)]).to_csv(OUT / "profiles_weekend_hourly.csv")

# ---------------------------------------------------------------- figures + numbers
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False}); hrs = np.arange(24)
fig, ax = plt.subplots(1, K, figsize=(3.2 * K, 3), sharey=True)
for c in range(K):
    m = (sites.cluster == c).values; sub = np.random.default_rng(0).choice(np.where(m)[0], size=min(60, m.sum()), replace=False)
    ax[c].plot(hrs, PWD[sub].T, color="0.85", lw=0.5); ax[c].plot(hrs, PWD[m].mean(0), color="C0", lw=2, label="weekday"); ax[c].plot(hrs, PWE[m].mean(0), color="C1", lw=2, ls="--", label="weekend")
    ax[c].set_title(f"{names[c]}\n(n={m.sum()})", fontsize=9); ax[c].set_xlabel("hour"); ax[c].set_xticks([0, 6, 12, 18, 24])
ax[0].set_ylabel("kW / weekday mean"); ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG / "archetype_profiles_v2.png"); plt.close()
ct = pd.crosstab(sites.building_type, sites.archetype); ax = ct.plot.barh(stacked=True, figsize=(7, 3.5)); ax.set_xlabel("building-years"); plt.tight_layout(); plt.savefig(FIG / "class_balance_by_type_v2.png"); plt.close()
ct2 = pd.crosstab(sites.source, sites.archetype); ax = ct2.plot.barh(stacked=True, figsize=(7, 2.2)); ax.set_xlabel("building-years"); plt.tight_layout(); plt.savefig(FIG / "class_balance_by_source_v2.png"); plt.close()
ct3 = pd.crosstab(sites.region, sites.archetype); ax = ct3.plot.barh(stacked=True, figsize=(7, 3)); ax.set_xlabel("building-years"); plt.tight_layout(); plt.savefig(FIG / "class_balance_by_region_v2.png"); plt.close()
num = dict(rows_kept=int(len(sites)), unique_buildings=int(sites.building_key.nunique()), by_source=sites.source.value_counts().to_dict(),
           duplicates_dropped=int((~sites_all.keep).sum()), ml_rows=int(len(X_wide)), ml_rows_aug=int(len(X_aug)), K=int(K),
           kmeans_check=dict(silhouette_k4=sil[4], adjusted_rand_vs_rules=round(float(ari), 3)), label_thresholds=dict(DN_FLAT=DN_FLAT, DN_DAY=DN_DAY, WE_7DAY=WE_7DAY),
           borderline_rows_margin_lt_0_1=int((sites.label_margin < 0.1).sum()), class_counts=sites.archetype.value_counts().to_dict(),
           class_by_source=ct2.to_dict(), type_counts=sites.building_type.value_counts().to_dict(), region_counts=sites.region.value_counts().to_dict(),
           sites_500kW_5MW=int(sites.peak_kw_hourly.between(500, 5000).sum()), enernoc_uplift_median=round(float(sites.interval_uplift.median()), 3))
json.dump(num, open(OUT / "numbers.json", "w"), indent=2, default=str); print(json.dumps(num, indent=1, default=str))
