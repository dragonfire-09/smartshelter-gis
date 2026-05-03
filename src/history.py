from pathlib import Path

import pandas as pd

from src.normalize import slugify_column


HISTORY_DIR = Path("data/history")
HISTORY_FILE = HISTORY_DIR / "shelter_history.csv"


HISTORY_COLUMNS = [
    "snapshot_date",
    "snapshot_ts",
    "source_name",
    "resource_label",
    "record_key",
    "name",
    "city",
    "district",
    "lat",
    "lon",
    "capacity",
    "occupancy",
    "vet_count",
    "sterilization_count",
    "adoption_count",
    "occupancy_rate",
    "animals_per_vet",
    "risk_score",
    "risk_level",
    "recommended_action",
    "coordinate_valid",
    "is_estimated",
    "data_quality_note",
]


def ensure_history_dir():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def make_record_key(row):
    """
    Aynı merkezin farklı günlerde eşleşebilmesi için stabil bir anahtar üretir.
    İdeal durumda bakanlık/belediye ID'si olmalı.
    Şimdilik city + district + name kombinasyonunu kullanıyoruz.
    """
    city = slugify_column(row.get("city", ""))
    district = slugify_column(row.get("district", ""))
    name = slugify_column(row.get("name", ""))

    key = f"{city}__{district}__{name}"

    if key.replace("_", "") == "":
        key = f"unknown__{row.name}"

    return key


def prepare_snapshot_df(df, source_name, resource_label):
    snapshot_df = df.copy()

    now = pd.Timestamp.now()

    snapshot_df["snapshot_date"] = now.strftime("%Y-%m-%d")
    snapshot_df["snapshot_ts"] = now.strftime("%Y-%m-%d %H:%M:%S")
    snapshot_df["source_name"] = source_name
    snapshot_df["resource_label"] = resource_label
    snapshot_df["record_key"] = snapshot_df.apply(make_record_key, axis=1)

    for col in HISTORY_COLUMNS:
        if col not in snapshot_df.columns:
            snapshot_df[col] = None

    snapshot_df = snapshot_df[HISTORY_COLUMNS]

    return snapshot_df


def load_history():
    ensure_history_dir()

    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        history_df = pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    for col in HISTORY_COLUMNS:
        if col not in history_df.columns:
            history_df[col] = None

    history_df = history_df[HISTORY_COLUMNS]

    numeric_cols = [
        "lat",
        "lon",
        "capacity",
        "occupancy",
        "vet_count",
        "sterilization_count",
        "adoption_count",
        "occupancy_rate",
        "animals_per_vet",
        "risk_score",
    ]

    for col in numeric_cols:
        history_df[col] = pd.to_numeric(history_df[col], errors="coerce")

    return history_df


def append_snapshot(df, source_name, resource_label):
    """
    Günlük snapshot kaydı oluşturur.

    Streamlit her etkileşimde tekrar çalıştığı için aynı gün + aynı kaynak + aynı kayıt için
    mükerrer veri oluşmasın diye eski satırı son veriyle değiştiriyoruz.
    """
    ensure_history_dir()

    if df.empty:
        return load_history()

    new_snapshot = prepare_snapshot_df(
        df=df,
        source_name=source_name,
        resource_label=resource_label,
    )

    old_history = load_history()

    combined = pd.concat(
        [old_history, new_snapshot],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=[
            "snapshot_date",
            "source_name",
            "resource_label",
            "record_key",
        ],
        keep="last",
    )

    combined = combined.sort_values(
        ["snapshot_date", "city", "district", "name"]
    )

    combined.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")

    return combined


def get_available_snapshot_dates(history_df):
    if history_df.empty or "snapshot_date" not in history_df.columns:
        return []

    dates = (
        history_df["snapshot_date"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    return dates


def build_history_summary(history_df):
    if history_df.empty:
        return pd.DataFrame(
            columns=[
                "snapshot_date",
                "record_count",
                "total_capacity",
                "total_occupancy",
                "avg_risk",
                "critical_count",
                "estimated_count",
            ]
        )

    summary = (
        history_df.groupby("snapshot_date", as_index=False)
        .agg(
            record_count=("record_key", "nunique"),
            total_capacity=("capacity", "sum"),
            total_occupancy=("occupancy", "sum"),
            avg_risk=("risk_score", "mean"),
            critical_count=(
                "risk_level",
                lambda s: (s.astype(str) == "Kritik").sum(),
            ),
            estimated_count=(
                "is_estimated",
                lambda s: (s.astype(str) == "True").sum(),
            ),
        )
    )

    summary["avg_risk"] = summary["avg_risk"].round(1)

    summary = summary.sort_values("snapshot_date")

    return summary


def compare_snapshot_dates(history_df, start_date, end_date):
    """
    İki tarih arasındaki merkez bazlı farkları hesaplar.
    """
    if history_df.empty:
        return pd.DataFrame()

    start_df = history_df[
        history_df["snapshot_date"].astype(str) == str(start_date)
    ].copy()

    end_df = history_df[
        history_df["snapshot_date"].astype(str) == str(end_date)
    ].copy()

    if start_df.empty or end_df.empty:
        return pd.DataFrame()

    keep_cols = [
        "record_key",
        "name",
        "city",
        "district",
        "capacity",
        "occupancy",
        "vet_count",
        "sterilization_count",
        "adoption_count",
        "occupancy_rate",
        "risk_score",
        "risk_level",
    ]

    start_df = start_df[keep_cols].add_suffix("_old")
    end_df = end_df[keep_cols].add_suffix("_new")

    merged = start_df.merge(
        end_df,
        left_on="record_key_old",
        right_on="record_key_new",
        how="outer",
    )

    merged["record_key"] = merged["record_key_new"].fillna(
        merged["record_key_old"]
    )

    merged["name"] = merged["name_new"].fillna(merged["name_old"])
    merged["city"] = merged["city_new"].fillna(merged["city_old"])
    merged["district"] = merged["district_new"].fillna(
        merged["district_old"]
    )

    numeric_pairs = [
        "capacity",
        "occupancy",
        "vet_count",
        "sterilization_count",
        "adoption_count",
        "occupancy_rate",
        "risk_score",
    ]

    for metric in numeric_pairs:
        old_col = f"{metric}_old"
        new_col = f"{metric}_new"
        delta_col = f"{metric}_delta"

        merged[delta_col] = merged[new_col].fillna(0) - merged[old_col].fillna(0)

    def status(row):
        if pd.isna(row.get("record_key_old")) and pd.notna(row.get("record_key_new")):
            return "Yeni kayıt"
        if pd.notna(row.get("record_key_old")) and pd.isna(row.get("record_key_new")):
            return "Kayıt artık yok"
        return "Devam eden kayıt"

    merged["change_status"] = merged.apply(status, axis=1)

    output_cols = [
        "change_status",
        "record_key",
        "name",
        "city",
        "district",
        "capacity_old",
        "capacity_new",
        "capacity_delta",
        "occupancy_old",
        "occupancy_new",
        "occupancy_delta",
        "occupancy_rate_old",
        "occupancy_rate_new",
        "occupancy_rate_delta",
        "risk_score_old",
        "risk_score_new",
        "risk_score_delta",
        "risk_level_old",
        "risk_level_new",
        "vet_count_old",
        "vet_count_new",
        "vet_count_delta",
        "sterilization_count_old",
        "sterilization_count_new",
        "sterilization_count_delta",
        "adoption_count_old",
        "adoption_count_new",
        "adoption_count_delta",
    ]

    for col in output_cols:
        if col not in merged.columns:
            merged[col] = None

    merged = merged[output_cols]

    merged = merged.sort_values(
        ["risk_score_delta", "occupancy_delta"],
        ascending=False,
    )

    return merged


def compare_summary(summary_df, start_date, end_date):
    if summary_df.empty:
        return {}

    old_row = summary_df[
        summary_df["snapshot_date"].astype(str) == str(start_date)
    ]

    new_row = summary_df[
        summary_df["snapshot_date"].astype(str) == str(end_date)
    ]

    if old_row.empty or new_row.empty:
        return {}

    old_row = old_row.iloc[0]
    new_row = new_row.iloc[0]

    metrics = [
        "record_count",
        "total_capacity",
        "total_occupancy",
        "avg_risk",
        "critical_count",
        "estimated_count",
    ]

    result = {}

    for metric in metrics:
        old_value = old_row.get(metric, 0)
        new_value = new_row.get(metric, 0)

        result[metric] = {
            "old": old_value,
            "new": new_value,
            "delta": new_value - old_value,
        }

    return result
