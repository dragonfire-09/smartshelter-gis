import re
import unicodedata

import pandas as pd


TURKEY_CITY_KEYWORDS = {
    "ibb": "İstanbul",
    "istanbul": "İstanbul",
    "b40": "İstanbul",
    "kadıköy": "İstanbul",
    "kadikoy": "İstanbul",
    "tuzla": "İstanbul",
    "izmir": "İzmir",
    "bizizmir": "İzmir",
    "kocaeli": "Kocaeli",
    "konya": "Konya",
    "ordu": "Ordu",
    "gaziantep": "Gaziantep",
}


ISTANBUL_DISTRICTS = {
    "ADALAR",
    "ARNAVUTKÖY",
    "ATAŞEHİR",
    "AVCILAR",
    "BAĞCILAR",
    "BAHÇELİEVLER",
    "BAKIRKÖY",
    "BAŞAKŞEHİR",
    "BAYRAMPAŞA",
    "BEŞİKTAŞ",
    "BEYKOZ",
    "BEYLİKDÜZÜ",
    "BEYOĞLU",
    "BÜYÜKÇEKMECE",
    "ÇATALCA",
    "ÇEKMEKÖY",
    "ESENLER",
    "ESENYURT",
    "EYÜPSULTAN",
    "FATİH",
    "GAZİOSMANPAŞA",
    "GÜNGÖREN",
    "KADIKÖY",
    "KAĞITHANE",
    "KARTAL",
    "KÜÇÜKÇEKMECE",
    "MALTEPE",
    "PENDİK",
    "SANCAKTEPE",
    "SARIYER",
    "SİLİVRİ",
    "SULTANBEYLİ",
    "SULTANGAZİ",
    "ŞİLE",
    "ŞİŞLİ",
    "TUZLA",
    "ÜMRANİYE",
    "ÜSKÜDAR",
    "ZEYTİNBURNU",
}


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
            "Yok": None,
            "yok": None,
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


def infer_city_from_text(text):
    text = str(text).lower()

    for key, city in TURKEY_CITY_KEYWORDS.items():
        if key in text:
            return city

    return ""


def infer_city_and_district(df):
    if "source_portal" not in df.columns:
        df["source_portal"] = ""

    if "source_resource" not in df.columns:
        df["source_resource"] = ""

    source_text = (
        df["source_portal"].astype(str)
        + " "
        + df["source_resource"].astype(str)
    )

    city_inferred = source_text.apply(infer_city_from_text)

    empty_city_mask = (
        df["city"].astype(str).str.strip().isin(["", "nan", "Belirtilmemiş"])
    )

    df.loc[empty_city_mask & city_inferred.ne(""), "city"] = city_inferred[
        empty_city_mask & city_inferred.ne("")
    ]

    # İlçe İstanbul ilçelerinden biriyse ve şehir boşsa İstanbul yap
    district_upper = df["district"].astype(str).str.upper().str.strip()

    ist_mask = district_upper.isin(ISTANBUL_DISTRICTS)

    df.loc[
        empty_city_mask & ist_mask,
        "city",
    ] = "İstanbul"

    return df


def build_name_from_source(df):
    default_name_mask = (
        df["name"].astype(str).str.strip().isin(
            ["", "nan", "Hayvan Bakımevi / Toplama Merkezi"]
        )
    )

    # İlçe varsa: "Kadıköy Hayvan Bakımevi / Barınak Verisi"
    district_ok = df["district"].astype(str).str.strip().ne("Belirtilmemiş")

    df.loc[
        default_name_mask & district_ok,
        "name",
    ] = (
        df.loc[default_name_mask & district_ok, "district"].astype(str)
        + " Hayvan Bakımevi / Barınak Verisi"
    )

    # Hala boşsa source_resource'tan üret
    default_name_mask = (
        df["name"].astype(str).str.strip().isin(
            ["", "nan", "Hayvan Bakımevi / Toplama Merkezi"]
        )
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
        "aciklama",
        "description",
    ]

    district_candidates = [
        "ilce",
        "ilce_adi",
        "district",
        "ilcesi",
        "ilce_ismi",
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

    capacity_candidates_contains = [
        "kapasite",
        "capacity",
        "kapasitesi",
        "animal_shelter_capacity",
        "shelter_capacity",
    ]

    shelter_count_candidates_contains = [
        "barinak_sayisi",
        "barınak_sayısı",
        "shelter_count",
        "number_of_shelter",
        "number_of_animal_shelter",
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

        elif any(k in c for k in capacity_candidates_contains):
            rename_map[c] = "capacity"

        elif any(k in c for k in shelter_count_candidates_contains):
            rename_map[c] = "shelter_count"

        elif (
            "doluluk" in c
            or "mevcut" in c
            or "hayvan_sayisi" in c
            or "hayvan_adedi" in c
            or "barinan_hayvan" in c
            or "occupancy" in c
            or "current_animal" in c
        ):
            rename_map[c] = "occupancy"

        elif "veteriner" in c or c == "vet" or "vet_count" in c:
            rename_map[c] = "vet_count"

        elif "kisir" in c or "kisirlastirma" in c or "sterilization" in c:
            rename_map[c] = "sterilization_count"

        elif "sahip" in c or "sahiplendirme" in c or "adoption" in c:
            rename_map[c] = "adoption_count"

        elif "adres" in c or "address" in c:
            rename_map[c] = "address"

        elif "telefon" in c or c == "tel" or "iletisim" in c or "phone" in c:
            rename_map[c] = "phone"

        elif "tarih" in c or "guncelleme" in c or "updated" in c or "year" == c or "yil" == c:
            rename_map[c] = "updated_at"

    df = df.rename(columns=rename_map)
    df = coalesce_duplicate_columns(df)

    required_defaults = {
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
        "shelter_count": None,
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

    for col, default in required_defaults.items():
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
        df[col] = df[col].fillna(required_defaults.get(col, "")).astype(str)
        df[col] = df[col].replace({"nan": required_defaults.get(col, "")})

    df = build_name_from_source(df)
    df = infer_city_and_district(df)

    numeric_cols = [
        "lat",
        "lon",
        "capacity",
        "occupancy",
        "vet_count",
        "sterilization_count",
        "adoption_count",
        "shelter_count",
        "relevance_score",
    ]

    for col in numeric_cols:
        df[col] = parse_numeric_series(df[col])

    # Gerçek alan var mı?
    df["capacity_available"] = df["capacity"].notna()
    df["occupancy_available"] = df["occupancy"].notna()
    df["vet_count_available"] = df["vet_count"].notna()
    df["sterilization_available"] = df["sterilization_count"].notna()
    df["adoption_available"] = df["adoption_count"].notna()
    df["shelter_count_available"] = df["shelter_count"].notna()

    # Geriye dönük uyumluluk için estimated kolonları
    df["capacity_estimated"] = ~df["capacity_available"]
    df["occupancy_estimated"] = ~df["occupancy_available"]
    df["vet_count_estimated"] = ~df["vet_count_available"]
    df["sterilization_count_estimated"] = ~df["sterilization_available"]
    df["adoption_count_estimated"] = ~df["adoption_available"]

    # Eksik sayısal alanları hesaplama kırılmasın diye 0 ile dolduruyoruz,
    # ama artık bunlar risk_ready değilse risk hesabına sokulmayacak.
    fill_values = {
        "capacity": 0,
        "occupancy": 0,
        "vet_count": 0,
        "sterilization_count": 0,
        "adoption_count": 0,
        "shelter_count": 0,
    }

    for col, val in fill_values.items():
        df[col] = df[col].fillna(val)

    df["coordinate_valid"] = (
        df["lat"].notna()
        & df["lon"].notna()
        & df["lat"].between(35, 43)
        & df["lon"].between(25, 46)
    )

    # Veri kapsamı sınıflandırması
    df["data_scope"] = "excluded"

    risk_ready_mask = (
        df["capacity_available"]
        & df["occupancy_available"]
        & df["capacity"].gt(0)
    )

    capacity_only_mask = (
        df["capacity_available"]
        & ~df["occupancy_available"]
        & df["capacity"].gt(0)
    )

    location_only_mask = (
        df["coordinate_valid"]
        & ~df["capacity_available"]
        & ~df["occupancy_available"]
    )

    stats_mask = (
        df["resource_category"].astype(str).eq("operation_stats")
        | df["shelter_count_available"]
    )

    df.loc[risk_ready_mask, "data_scope"] = "risk_ready"
    df.loc[capacity_only_mask, "data_scope"] = "capacity_only"
    df.loc[location_only_mask, "data_scope"] = "location_only"
    df.loc[stats_mask & ~risk_ready_mask & ~capacity_only_mask, "data_scope"] = "operation_stats"

    df["analytics_eligible"] = df["data_scope"].isin(
        ["risk_ready", "capacity_only", "location_only"]
    )

    df["risk_eligible"] = df["data_scope"].eq("risk_ready")

    df["analytics_exclusion_reason"] = ""

    df.loc[
        df["data_scope"].eq("excluded"),
        "analytics_exclusion_reason",
    ] = "kapasite, mevcut hayvan veya konum verisi ana analitik için yeterli değil; "

    df.loc[
        df["city"].astype(str).str.strip().isin(["", "nan", "Belirtilmemiş"]),
        "data_quality_note",
    ] += "il bilgisi kaynakta yok veya çıkarılamadı; "

    df.loc[
        df["district"].astype(str).str.strip().isin(["", "nan", "Belirtilmemiş"]),
        "data_quality_note",
    ] += "ilçe bilgisi kaynakta yok; "

    df.loc[
        ~df["capacity_available"],
        "data_quality_note",
    ] += "kapasite alanı kaynakta yok; "

    df.loc[
        ~df["occupancy_available"],
        "data_quality_note",
    ] += "mevcut hayvan alanı kaynakta yok; "

    df.loc[
        ~df["coordinate_valid"],
        "data_quality_note",
    ] += "koordinat eksik/geçersiz; "

    df["is_estimated"] = df["data_quality_note"].str.strip().ne("")

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
        "shelter_count",
        "address",
        "phone",
        "updated_at",
        "source_portal",
        "source_resource",
        "source_url",
        "resource_category",
        "relevance_score",
        "capacity_available",
        "occupancy_available",
        "vet_count_available",
        "sterilization_available",
        "adoption_available",
        "shelter_count_available",
        "capacity_estimated",
        "occupancy_estimated",
        "vet_count_estimated",
        "sterilization_count_estimated",
        "adoption_count_estimated",
        "coordinate_valid",
        "data_scope",
        "analytics_eligible",
        "risk_eligible",
        "analytics_exclusion_reason",
        "data_quality_note",
        "is_estimated",
    ]

    return pd.DataFrame(columns=columns)
