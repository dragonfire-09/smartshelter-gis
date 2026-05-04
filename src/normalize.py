import re
import pandas as pd


def slugify_column(col: str) -> str:
    if not isinstance(col, str):
        return ""

    s = col.strip().lower()
    s = (
        s.replace("ı", "i")
        .replace("İ", "i")
        .replace("ş", "s")
        .replace("Ş", "s")
        .replace("ğ", "g")
        .replace("Ğ", "g")
        .replace("ü", "u")
        .replace("Ü", "u")
        .replace("ö", "o")
        .replace("Ö", "o")
        .replace("ç", "c")
        .replace("Ç", "c")
    )
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _canonical_key(col: str) -> str:
    return slugify_column(col)


COLUMN_ALIASES = {
    "name": [
        "name", "ad", "adi", "ad_", "merkez_adi", "tesis_adi",
        "barinak_adi", "kurum_adi", "kuruluş", "kurulus", "tesis",
    ],
    "city": [
        "city", "il", "il_adi", "sehir", "şehir", "il_ad",
    ],
    "district": [
        "district", "ilce", "ilce_adi", "ilçe", "ilçe_adi",
    ],
    "capacity": [
        "capacity", "kapasite", "barinak_kapasitesi", "kapasite_sayisi",
    ],
    "occupancy": [
        "occupancy", "mevcut", "mevcut_hayvan", "hayvan_sayisi",
        "barinaktaki_hayvan_sayisi", "mevcut_hayvan_sayisi",
    ],
    "vet_count": [
        "vet_count", "veteriner_sayisi", "veteriner",
        "veteriner_hekim_sayisi", "veteriner_hekim",
    ],
    "sterilization_count": [
        "sterilization_count", "kisirlastirma", "kisirlastirma_sayisi",
        "kısırlaştırma", "kisirlastirilan",
    ],
    "adoption_count": [
        "adoption_count", "sahiplendirme", "sahiplendirme_sayisi",
        "sahiplendirilen",
    ],
    "latitude": ["latitude", "lat", "enlem", "y"],
    "longitude": ["longitude", "lon", "lng", "boylam", "x"],
}


def _detect_columns(df: pd.DataFrame) -> dict:
    mapping = {}
    available = {_canonical_key(c): c for c in df.columns}

    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = _canonical_key(alias)
            if key in available:
                mapping[target] = available[key]
                break

    return mapping


def _classify_resource_category(row) -> str:
    parts = " ".join(
        str(row.get(c, "")) for c in ["name", "package", "matched_query", "source_resource"]
    ).lower()

    irrelevant_keywords = [
        "vektor", "vektör", "ilaclama", "ilaçlama", "haşere",
        "sivrisinek", "fare", "kemirgen", "böcek", "bocek",
    ]
    if any(k in parts for k in irrelevant_keywords):
        return "irrelevant"

    stats_keywords = [
        "istatistik", "yıllara", "yillara", "evcil hayvan varlığı",
        "evcil hayvan varligi", "hane", "denetim", "uygulama sayilari",
        "uygulama sayıları",
    ]
    if any(k in parts for k in stats_keywords):
        return "operation_stats"

    health_keywords = [
        "sağlık kurum", "saglik kurum", "sağlık tesis", "saglik tesis",
        "hastane", "kuruluş",
    ]
    if any(k in parts for k in health_keywords):
        return "general_health"

    shelter_keywords = [
        "barınak", "barinak", "bakimevi", "bakımevi",
        "geçici hayvan bakım", "gecici hayvan bakim",
        "rehabilitasyon",
    ]
    if any(k in parts for k in shelter_keywords):
        return "shelter_facility"

    return "unknown"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    mapping = _detect_columns(df)

    target_columns = list(COLUMN_ALIASES.keys())

    out = pd.DataFrame(index=df.index)

    for target in target_columns:
        if target in mapping:
            out[target] = df[mapping[target]]
        else:
            out[target] = pd.NA

    for meta in [
        "source_portal",
        "source_resource",
        "source_url",
        "resource_category",
        "relevance_score",
        "package",
        "matched_query",
    ]:
        if meta in df.columns:
            out[meta] = df[meta]

    numeric_cols = [
        "capacity",
        "occupancy",
        "vet_count",
        "sterilization_count",
        "adoption_count",
        "latitude",
        "longitude",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["name_available"] = out["name"].notna() & out["name"].astype(str).str.strip().ne("")
    out["city_available"] = out["city"].notna() & out["city"].astype(str).str.strip().ne("")
    out["capacity_available"] = out["capacity"].notna() & (out["capacity"] > 0)
    out["occupancy_available"] = out["occupancy"].notna() & (out["occupancy"] >= 0)
    out["vet_count_available"] = out["vet_count"].notna() & (out["vet_count"] >= 0)

    lat_ok = out["latitude"].between(-90, 90)
    lon_ok = out["longitude"].between(-180, 180)
    out["coordinate_valid"] = lat_ok & lon_ok

    if "resource_category" not in out.columns or out["resource_category"].isna().all():
        out["resource_category"] = out.apply(_classify_resource_category, axis=1)
    else:
        empty_mask = out["resource_category"].isna() | out["resource_category"].astype(str).str.strip().eq("")
        if empty_mask.any():
            out.loc[empty_mask, "resource_category"] = out.loc[empty_mask].apply(
                _classify_resource_category, axis=1
            )

    def _scope(row):
        cat = str(row.get("resource_category", "")).lower()
        if cat in ["irrelevant", "operation_stats", "general_health"]:
            return "operation_stats"

        if row.get("capacity_available") and row.get("occupancy_available") and row.get("name_available"):
            return "risk_ready"

        if row.get("capacity_available") and row.get("name_available"):
            return "capacity_only"

        if row.get("coordinate_valid") and row.get("name_available"):
            return "location_only"

        return "unknown"

    out["data_scope"] = out.apply(_scope, axis=1)

    out["risk_eligible"] = out["data_scope"].eq("risk_ready")

    out["analytics_eligible"] = out["data_scope"].isin(
        ["risk_ready", "capacity_only", "location_only"]
    ) & out["name_available"]

    def _exclusion_reason(row):
        reasons = []

        if not row.get("name_available"):
            reasons.append("isim alanı yok")

        cat = str(row.get("resource_category", "")).lower()
        if cat == "irrelevant":
            reasons.append("ilgisiz kaynak (vektör/haşere ilaçlama vb.)")
        if cat == "operation_stats":
            reasons.append("operasyonel istatistik kaynağı")
        if cat == "general_health":
            reasons.append("genel sağlık kurum kaynağı (barınak değil)")

        if not row.get("capacity_available") and not row.get("coordinate_valid"):
            reasons.append("kapasite ve koordinat yok")

        return "; ".join(reasons)

    out["analytics_exclusion_reason"] = out.apply(
        lambda r: _exclusion_reason(r) if not r["analytics_eligible"] else "",
        axis=1,
    )

    out["occupancy_rate"] = pd.NA
    valid_op = out["capacity_available"] & out["occupancy_available"]
    out.loc[valid_op, "occupancy_rate"] = (
        (out.loc[valid_op, "occupancy"] / out.loc[valid_op, "capacity"]) * 100
    ).round(1)

    out["animals_per_vet"] = pd.NA
    valid_vet = out["occupancy_available"] & out["vet_count_available"] & (out["vet_count"] > 0)
    out.loc[valid_vet, "animals_per_vet"] = (
        out.loc[valid_vet, "occupancy"] / out.loc[valid_vet, "vet_count"]
    ).round(1)

    def _quality_note(row):
        notes = []
        for col_label, col_flag in [
            ("name", "name_available"),
            ("city", "city_available"),
            ("capacity", "capacity_available"),
            ("occupancy", "occupancy_available"),
            ("vet_count", "vet_count_available"),
        ]:
            if not row.get(col_flag):
                notes.append(f"{col_label} alanı kaynakta yok")

        if not row.get("coordinate_valid"):
            notes.append("koordinat eksik/geçersiz")

        return "; ".join(notes)

    out["data_quality_note"] = out.apply(_quality_note, axis=1)

    return out
