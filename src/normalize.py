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


def infer_location_from_source(df):
    """
    Kaynak portal adından il/ilçe tahmini.
    Bu yalnızca kaynakta il/ilçe yoksa devreye girer.
    """

    if "source_portal" not in df.columns:
        return df

    portal = df["source_portal"].astype(str).str.lower()

    city_map = {
        "ibb": "İstanbul",
        "b40": "İstanbul",
        "kadıköy": "İstanbul",
        "kadikoy": "İstanbul",
        "tuzla": "İstanbul",
        "izmir": "İzmir",
        "kocaeli": "Kocaeli",
        "konya": "Konya",
        "ordu": "Ordu",
        "gaziantep": "Gaziantep",
    }

    district_map = {
        "kadıköy": "Kadıköy",
        "kadikoy": "Kadıköy",
        "tuzla": "Tuzla",
    }

    for key, city in city_map.items():
        mask = portal.str.contains(key, na=False)
        df.loc[
            mask & df["city"].astype(str).str.strip().isin(["", "nan", "Belirtilmemiş"]),
            "city",
        ] = city

    for key, district in district_map.items():
        mask = portal.str.contains(key, na=False)
        df.loc[
            mask & df["district"].astype(str).str.strip().isin(["", "nan", "Belirtilmemiş"]),
            "district",
        ] = district

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
        "facility_name",
        "name",
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

        elif "kapasite" in c or "capacity" in c:
            rename_map[c] = "capacity"

        elif (
            "doluluk" in c
            or "mevcut" in c
            or "hayvan_sayisi" in c
            or "hayvan_adedi" in c
            or "barinan_hayvan" in c
            or "occupancy" in c
        ):
            rename_map[c] = "occupancy"

        elif "veteriner" in c or "vet" == c or "vet_count" in c:
            rename_map[c] = "vet_count"

        elif "kisir" in c or "kisirlastirma" in c or "sterilization" in c:
            rename_map[c] = "sterilization_count"

        elif "sahip" in c or "sahiplendirme" in c or "adoption" in c:
            rename_map[c] = "adoption_count"

        elif "adres" in c or "address" in c:
            rename_map[c] = "address"

        elif "telefon" in c or c == "tel" or "iletisim" in c or "phone" in c:
            rename_map[c] = "phone"

        elif "tarih" in c or "guncelleme" in c or "updated" in c:
            rename_map[c] = "updated_at"

    df = df.rename(columns=rename_map)
    df = coalesce_duplicate_columns(df)

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

    optional_defaults = {
        "address": "",
        "phone": "",
        "updated_at": "",
        "source_portal": "",
        "source_resource": "",
        "source_url": "",
        "resource_category": "",
        "relevance_score": 0,
    }

    df["data_quality_note"] = ""

    for col, default in critical_defaults.items():
        if col not in df.columns:
            df[col] = default

            # Sadece ana kimlik ve temel metriklerde not üret.
            if col in ["name", "city", "district", "capacity", "occupancy", "vet_count"]:
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
        "source_portal",
        "source_resource",
        "source_url",
        "resource_category",
    ]

    for col in text_cols:
        default = critical_defaults.get(col, optional_defaults.get(col, ""))
        df[col] = df[col].fillna(default).astype(str)
        df[col] = df[col].replace({"nan": default})

    # Eğer name yoksa ama source_resource varsa, kaynak adından daha anlamlı ad üret.
    default_name_mask = (
        df["name"].astype(str).str.strip()
        == "Hayvan Bakımevi / Toplama Merkezi"
    )

    has_source_resource = df["source_resource"].astype(str).str.strip().ne("")

    df.loc[
        default_name_mask & has_source_resource,
        "name",
    ] = (
        df.loc[default_name_mask & has_source_resource, "source_resource"]
        .astype(str)
        .str.split("|")
        .str[-1]
        .str.replace(".csv", "", regex=False)
        .str.replace(".xlsx", "", regex=False)
        .str.replace(".xls", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.strip()
    )

    df = infer_location_from_source(df)

    numeric_cols = [
        "lat",
        "lon",
        "capacity",
        "occupancy",
        "vet_count",
        "sterilization_count",
        "adoption_count",
        "relevance_score",
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

        # Kısırlaştırma/sahiplendirme eksikliği çok yaygın olduğu için kalite notunu şişirmeyelim.
        if col in ["capacity", "occupancy", "vet_count"]:
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

    # Ana risk dashboard'una uygunluk kontrolü
    df["analytics_eligible"] = True
    df["analytics_exclusion_reason"] = ""

    if "resource_category" in df.columns:
        cat = df["resource_category"].astype(str)
        category_known_mask = cat.str.strip().ne("")
        not_facility_mask = category_known_mask & cat.ne("shelter_facility")

        df.loc[not_facility_mask, "analytics_eligible"] = False
        df.loc[not_facility_mask, "analytics_exclusion_reason"] += (
            "resource barınak/bakımevi envanteri değil; "
        )

    missing_core_metrics_mask = (
        df["capacity_estimated"].astype(bool)
        & df["occupancy_estimated"].astype(bool)
    )

    df.loc[missing_core_metrics_mask, "analytics_eligible"] = False
    df.loc[missing_core_metrics_mask, "analytics_exclusion_reason"] += (
        "kapasite ve mevcut hayvan alanları kaynakta yok; "
    )

    default_or_empty_name_mask = (
        df["name"].astype(str).str.strip().isin(
            ["", "nan", "Hayvan Bakımevi / Toplama Merkezi"]
        )
    )

    df.loc[default_or_empty_name_mask, "analytics_eligible"] = False
    df.loc[default_or_empty_name_mask, "analytics_exclusion_reason"] += (
        "merkez adı kaynakta yok; "
    )

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
        "resource_category",
        "relevance_score",
        "data_quality_note",
        "capacity_estimated",
        "occupancy_estimated",
        "vet_count_estimated",
        "sterilization_count_estimated",
        "adoption_count_estimated",
        "coordinate_valid",
        "is_estimated",
        "analytics_eligible",
        "analytics_exclusion_reason",
    ]

    return pd.DataFrame(columns=columns)
