import re
import unicodedata

import pandas as pd


def slugify_column(col):
    col = str(col).strip().lower()
    col = unicodedata.normalize("NFKD", col)
    col = col.encode("ascii", "ignore").decode("ascii")
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = col.strip("_")
    return col


def parse_numeric_series(series):
    s = series.astype(str).str.strip()

    s = s.replace(
        {
            "": None,
            "nan": None,
            "None": None,
            "NONE": None,
            "-": None,
        }
    )

    comma_mask = s.str.contains(",", regex=False, na=False)

    s.loc[comma_mask] = (
        s.loc[comma_mask]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    return pd.to_numeric(s, errors="coerce")


def coalesce_duplicate_columns(df):
    duplicated_names = df.columns[df.columns.duplicated()].unique().tolist()

    for col in duplicated_names:
        same_cols = df.loc[:, df.columns == col]
        combined = same_cols.bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=same_cols.columns)
        df[col] = combined

    return df


def normalize_columns(df):
    df = df.copy()

    if df.empty:
        return create_empty_normalized_df()

    df.columns = [slugify_column(c) for c in df.columns]

    rename_map = {}

    name_candidates = [
        "adi",
        "ad",
        "isim",
        "tesis_adi",
        "merkez_adi",
        "barinak_adi",
        "bakimevi_adi",
        "hayvan_bakimevi_adi",
        "kurum_adi",
        "birim_adi",
        "tesis",
        "merkez",
        "barinak",
        "bakimevi",
    ]

    district_candidates = [
        "ilce",
        "ilce_adi",
        "district",
        "ilcesi",
    ]

    city_candidates = [
        "il",
        "il_adi",
        "city",
        "sehir",
        "province",
    ]

    lat_candidates = [
        "enlem",
        "lat",
        "latitude",
        "y",
        "koordinat_y",
        "koordinat_enlem",
    ]

    lon_candidates = [
        "boylam",
        "lon",
        "lng",
        "longitude",
        "x",
        "koordinat_x",
        "koordinat_boylam",
    ]

    for c in df.columns:
        if c in name_candidates:
            rename_map[c] = "name"

        elif c in district_candidates:
            rename_map[c] = "district"

        elif c in city_candidates:
            rename_map[c] = "city"

        elif c in lat_candidates:
            rename_map[c] = "lat"

        elif c in lon_candidates:
            rename_map[c] = "lon"

        elif "kapasite" in c:
            rename_map[c] = "capacity"

        elif (
            "doluluk" in c
            or "mevcut" in c
            or "hayvan_sayisi" in c
            or "hayvan_adedi" in c
            or "barinan_hayvan" in c
        ):
            rename_map[c] = "occupancy"

        elif "veteriner" in c:
            rename_map[c] = "vet_count"

        elif "kisir" in c or "kisirlastirma" in c:
            rename_map[c] = "sterilization_count"

        elif "sahip" in c or "sahiplendirme" in c:
            rename_map[c] = "adoption_count"

        elif "adres" in c:
            rename_map[c] = "address"

        elif "telefon" in c or c == "tel" or "iletisim" in c:
            rename_map[c] = "phone"

        elif "tarih" in c or "guncelleme" in c or "updated" in c:
            rename_map[c] = "updated_at"

    df = df.rename(columns=rename_map)
    df = coalesce_duplicate_columns(df)

    # Kritik alanlar: kalite notuna yazılır.
    critical_defaults = {
        "name": "Hayvan Bakımevi / Toplama Merkezi",
        "city": "Belirtilmemiş",
        "district": "Belirtilmemiş",
        "lat": None,
        "lon": None,
        "capacity": None,
        "occupancy": None,
        "vet_count": None,
        "sterilization_count": None,
        "adoption_count": None,
    }

    # Opsiyonel alanlar: eksikse kalite problemi sayılmaz.
    optional_defaults = {
        "address": "",
        "phone": "",
        "updated_at": "",
    }

    df["data_quality_note"] = ""

    for col, default in critical_defaults.items():
        if col not in df.columns:
            df[col] = default
            df["data_quality_note"] += f"{col} alanı kaynakta yok; "

    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    text_cols = [
        "name",
        "city",
        "district",
        "address",
        "phone",
        "updated_at",
    ]

    for col in text_cols:
        default = critical_defaults.get(col, optional_defaults.get(col, ""))
        df[col] = df[col].fillna(default).astype(str)
        df[col] = df[col].replace({"nan": default})

    numeric_cols = [
        "lat",
        "lon",
        "capacity",
        "occupancy",
        "vet_count",
        "sterilization_count",
        "adoption_count",
    ]

    for col in numeric_cols:
        df[col] = parse_numeric_series(df[col])

    fallback_values = {
        "capacity": 100,
        "occupancy": 70,
        "vet_count": 1,
        "sterilization_count": 0,
        "adoption_count": 0,
    }

    for col, fallback in fallback_values.items():
        estimated_col = f"{col}_estimated"
        df[estimated_col] = df[col].isna()

        missing_mask = df[col].isna()
        df.loc[missing_mask, col] = fallback
        df.loc[missing_mask, "data_quality_note"] += (
            f"{col} tahmini değerle tamamlandı; "
        )

    df["coordinate_valid"] = (
        df["lat"].notna()
        & df["lon"].notna()
        & df["lat"].between(35, 43)
        & df["lon"].between(25, 46)
    )

    df.loc[
        df["coordinate_valid"] == False,  # noqa: E712
        "data_quality_note",
    ] += "koordinat eksik/geçersiz; "

    df["is_estimated"] = df["data_quality_note"].str.strip().ne("")

    df["capacity"] = df["capacity"].clip(lower=1)
    df["occupancy"] = df["occupancy"].clip(lower=0)
    df["vet_count"] = df["vet_count"].clip(lower=0)
    df["sterilization_count"] = df["sterilization_count"].clip(lower=0)
    df["adoption_count"] = df["adoption_count"].clip(lower=0)

    # Türkiye geneli taramada kaynak takibi için kolonlar yoksa oluştur.
    if "source_portal" not in df.columns:
        df["source_portal"] = ""

    if "source_resource" not in df.columns:
        df["source_resource"] = ""

    if "source_url" not in df.columns:
        df["source_url"] = ""

    return df


def create_empty_normalized_df():
    columns = [
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
        "address",
        "phone",
        "updated_at",
        "source_portal",
        "source_resource",
        "source_url",
        "data_quality_note",
        "capacity_estimated",
        "occupancy_estimated",
        "vet_count_estimated",
        "sterilization_count_estimated",
        "adoption_count_estimated",
        "coordinate_valid",
        "is_estimated",
    ]

    return pd.DataFrame(columns=columns)
