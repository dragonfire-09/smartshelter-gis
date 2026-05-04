from __future__ import annotations

import os
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests


DEMO_CSV_PATH = "data/kocaeli_shelters.csv"

DEFAULT_CKAN_PORTALS = [
    # İstersen Streamlit Secrets içinden CKAN_PORTALS ile override et.
    # Örnek secrets:
    # CKAN_PORTALS = "https://data.ibb.gov.tr,https://acikveri.kocaeli.bel.tr"
    "https://data.ibb.gov.tr",
    "https://acikveri.kocaeli.bel.tr",
]

CKAN_KEYWORDS = [
    "hayvan barınağı",
    "hayvan barinagi",
    "barınak",
    "barinak",
    "sokak hayvanları",
    "sokak hayvanlari",
    "veteriner",
    "geçici bakımevi",
    "gecici bakimevi",
]

SUPPORTED_FORMATS = {"csv", "xlsx", "xls", "json", "geojson"}


@dataclass
class DataLoadResult:
    df: pd.DataFrame
    source_label: str
    mode: str
    is_demo: bool = False
    errors: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


def _get_secret(name: str, default: str = "") -> str:
    try:
        import streamlit as st

        value = st.secrets.get(name, default)
        return str(value).strip() if value is not None else default
    except Exception:
        return os.getenv(name, default).strip()


def _get_ckan_portals() -> list[str]:
    raw = _get_secret("CKAN_PORTALS", "")
    if raw:
        portals = [x.strip().rstrip("/") for x in raw.split(",") if x.strip()]
        if portals:
            return portals

    return [x.rstrip("/") for x in DEFAULT_CKAN_PORTALS]


def _clean_col_name(col: Any) -> str:
    s = str(col).strip().lower()
    tr = str.maketrans("çğıöşüİ", "cgiosui")
    s = s.translate(tr)
    s = s.replace("\n", " ").replace("\r", " ")
    s = " ".join(s.split())
    return s


def _first_existing(df: pd.DataFrame, aliases: list[str]) -> str | None:
    cleaned = {_clean_col_name(c): c for c in df.columns}
    for alias in aliases:
        key = _clean_col_name(alias)
        if key in cleaned:
            return cleaned[key]
    return None


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False),
        errors="coerce",
    )


def _risk_level(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "Veri yetersiz"
    if score >= 80:
        return "Kritik"
    if score >= 60:
        return "Yüksek"
    if score >= 35:
        return "Orta"
    return "Düşük"


def normalize_shelter_df(
    df: pd.DataFrame,
    source_portal: str = "",
    source_dataset: str = "",
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    name_col = _first_existing(
        df,
        [
            "name",
            "ad",
            "adi",
            "adı",
            "barinak adi",
            "barınak adı",
            "tesis adi",
            "tesis adı",
            "kurum adi",
            "kurum adı",
        ],
    )

    city_col = _first_existing(
        df,
        [
            "city",
            "il",
            "sehir",
            "şehir",
            "province",
        ],
    )

    district_col = _first_existing(
        df,
        [
            "district",
            "ilce",
            "ilçe",
            "county",
        ],
    )

    lat_col = _first_existing(
        df,
        [
            "lat",
            "latitude",
            "enlem",
            "y",
            "koordinat y",
        ],
    )

    lon_col = _first_existing(
        df,
        [
            "lon",
            "lng",
            "long",
            "longitude",
            "boylam",
            "x",
            "koordinat x",
        ],
    )

    capacity_col = _first_existing(
        df,
        [
            "capacity",
            "kapasite",
            "hayvan kapasitesi",
            "toplam kapasite",
        ],
    )

    occupancy_col = _first_existing(
        df,
        [
            "occupancy",
            "mevcut",
            "mevcut hayvan",
            "hayvan sayisi",
            "hayvan sayısı",
            "sayi",
            "sayı",
        ],
    )

    risk_score_col = _first_existing(
        df,
        [
            "risk_score",
            "risk skoru",
            "risk puani",
            "risk puanı",
        ],
    )

    risk_level_col = _first_existing(
        df,
        [
            "risk_level",
            "risk seviyesi",
            "risk",
        ],
    )

    out = pd.DataFrame()

    if name_col:
        out["name"] = df[name_col].astype(str).str.strip()
    else:
        out["name"] = "İsimsiz Barınak / Tesis"

    if city_col:
        out["city"] = df[city_col].astype(str).str.strip()
    else:
        out["city"] = ""

    if district_col:
        out["district"] = df[district_col].astype(str).str.strip()
    else:
        out["district"] = ""

    if lat_col:
        out["lat"] = _to_num(df[lat_col])
    else:
        out["lat"] = pd.NA

    if lon_col:
        out["lon"] = _to_num(df[lon_col])
    else:
        out["lon"] = pd.NA

    if capacity_col:
        out["capacity"] = _to_num(df[capacity_col])
    else:
        out["capacity"] = pd.NA

    if occupancy_col:
        out["occupancy"] = _to_num(df[occupancy_col])
    else:
        out["occupancy"] = pd.NA

    if risk_score_col:
        out["risk_score"] = _to_num(df[risk_score_col])
    else:
        cap = pd.to_numeric(out["capacity"], errors="coerce")
        occ = pd.to_numeric(out["occupancy"], errors="coerce")

        ratio = occ / cap
        score = ratio * 70
        score = score.where(cap.notna() & occ.notna() & (cap > 0), pd.NA)
        out["risk_score"] = score.clip(lower=0, upper=100)

    if risk_level_col:
        out["risk_level"] = df[risk_level_col].astype(str).str.strip()
        out["risk_level"] = out["risk_level"].replace(
            {
                "nan": "Veri yetersiz",
                "None": "Veri yetersiz",
                "": "Veri yetersiz",
            }
        )
    else:
        out["risk_level"] = out["risk_score"].apply(_risk_level)

    out["source_portal"] = source_portal or "Bilinmeyen kaynak"
    out["source_dataset"] = source_dataset or ""

    out["name"] = out["name"].replace(
        {
            "nan": "İsimsiz Barınak / Tesis",
            "None": "İsimsiz Barınak / Tesis",
            "": "İsimsiz Barınak / Tesis",
        }
    )

    valid_coordinate = (
        pd.to_numeric(out["lat"], errors="coerce").between(-90, 90)
        & pd.to_numeric(out["lon"], errors="coerce").between(-180, 180)
    )

    # Koordinatsız veriyi tamamen atma.
    # Dashboardda görülebilsin ama haritada marker olmaz.
    out["has_valid_coordinate"] = valid_coordinate

    return out


def load_demo_data(path: str = DEMO_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_shelter_df(
        df,
        source_portal=f"Stabil Demo CSV · {path}",
        source_dataset=os.path.basename(path),
    )


def _ckan_action_url(portal: str, action: str) -> str:
    portal = portal.rstrip("/") + "/"
    return urljoin(portal, f"api/3/action/{action}")


def _safe_get_json(url: str, params: dict[str, Any] | None = None, timeout: int = 20):
    r = requests.get(
        url,
        params=params or {},
        timeout=timeout,
        headers={
            "User-Agent": "SmartShelter-GIS/1.0",
            "Accept": "application/json",
        },
    )
    r.raise_for_status()
    data = r.json()

    if not data.get("success", False):
        raise RuntimeError(data.get("error", "CKAN success=false"))

    return data.get("result")


def _search_packages(portal: str, keyword: str, rows: int = 10) -> list[dict[str, Any]]:
    url = _ckan_action_url(portal, "package_search")
    result = _safe_get_json(
        url,
        params={
            "q": keyword,
            "rows": rows,
        },
    )
    return result.get("results", []) if isinstance(result, dict) else []


def _read_resource_from_datastore(portal: str, resource_id: str) -> pd.DataFrame:
    url = _ckan_action_url(portal, "datastore_search")
    result = _safe_get_json(
        url,
        params={
            "resource_id": resource_id,
            "limit": 5000,
        },
    )

    records = result.get("records", []) if isinstance(result, dict) else []
    return pd.DataFrame(records)


def _read_resource_url(url: str, fmt: str) -> pd.DataFrame:
    fmt = (fmt or "").lower().strip()

    r = requests.get(
        url,
        timeout=35,
        headers={
            "User-Agent": "SmartShelter-GIS/1.0",
        },
    )
    r.raise_for_status()

    content = r.content

    if fmt == "csv" or url.lower().split("?")[0].endswith(".csv"):
        text = content.decode("utf-8-sig", errors="replace")
        try:
            return pd.read_csv(StringIO(text))
        except Exception:
            return pd.read_csv(StringIO(text), sep=";")

    if fmt in ("xlsx", "xls") or url.lower().split("?")[0].endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(content))

    if fmt in ("json", "geojson") or url.lower().split("?")[0].endswith((".json", ".geojson")):
        data = r.json()

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            if "features" in data and isinstance(data["features"], list):
                rows = []
                for f in data["features"]:
                    props = f.get("properties", {}) or {}
                    geom = f.get("geometry", {}) or {}
                    coords = geom.get("coordinates", [])

                    if (
                        isinstance(coords, list)
                        and len(coords) >= 2
                        and geom.get("type") == "Point"
                    ):
                        props["lon"] = coords[0]
                        props["lat"] = coords[1]

                    rows.append(props)

                return pd.DataFrame(rows)

            for key in ("records", "result", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])

            return pd.DataFrame([data])

    return pd.DataFrame()


def _resource_format(resource: dict[str, Any]) -> str:
    fmt = str(resource.get("format") or "").lower().strip()
    if fmt:
        return fmt

    url = str(resource.get("url") or "").lower().split("?")[0]
    for ext in SUPPORTED_FORMATS:
        if url.endswith(f".{ext}"):
            return ext

    return ""


def _resource_is_supported(resource: dict[str, Any]) -> bool:
    fmt = _resource_format(resource)
    url = str(resource.get("url") or "")
    return bool(url) and fmt in SUPPORTED_FORMATS


def load_ckan_data(
    portals: list[str] | None = None,
    keywords: list[str] | None = None,
) -> DataLoadResult:
    portals = portals or _get_ckan_portals()
    keywords = keywords or CKAN_KEYWORDS

    all_frames: list[pd.DataFrame] = []
    errors: list[str] = []
    debug = {
        "portals": portals,
        "keywords": keywords,
        "packages_found": 0,
        "resources_read": 0,
        "rows_before_normalize": 0,
        "rows_after_normalize": 0,
    }

    seen_resources: set[str] = set()

    for portal in portals:
        for keyword in keywords:
            try:
                packages = _search_packages(portal, keyword)
                debug["packages_found"] += len(packages)
            except Exception as e:
                errors.append(f"{portal} · '{keyword}' araması başarısız: {e}")
                continue

            for package in packages:
                package_title = package.get("title") or package.get("name") or ""
                resources = package.get("resources", []) or []

                for resource in resources:
                    resource_id = str(resource.get("id") or "")
                    resource_url = str(resource.get("url") or "")
                    resource_key = resource_id or resource_url

                    if not resource_key or resource_key in seen_resources:
                        continue

                    seen_resources.add(resource_key)

                    if not _resource_is_supported(resource):
                        continue

                    fmt = _resource_format(resource)

                    try:
                        raw_df = pd.DataFrame()

                        if resource.get("datastore_active") and resource_id:
                            try:
                                raw_df = _read_resource_from_datastore(portal, resource_id)
                            except Exception:
                                raw_df = pd.DataFrame()

                        if raw_df.empty and resource_url:
                            raw_df = _read_resource_url(resource_url, fmt)

                        if raw_df.empty:
                            continue

                        debug["resources_read"] += 1
                        debug["rows_before_normalize"] += len(raw_df)

                        norm = normalize_shelter_df(
                            raw_df,
                            source_portal=portal,
                            source_dataset=package_title,
                        )

                        if not norm.empty:
                            all_frames.append(norm)

                    except Exception as e:
                        errors.append(
                            f"{portal} · {package_title} · kaynak okunamadı: {e}"
                        )

    if not all_frames:
        return DataLoadResult(
            df=pd.DataFrame(),
            source_label="Türkiye Geneli CKAN Taraması · veri alınamadı",
            mode="Türkiye Geneli CKAN Taraması",
            is_demo=False,
            errors=errors or ["CKAN taramasında uygun veri bulunamadı."],
            debug=debug,
        )

    df = pd.concat(all_frames, ignore_index=True)

    # Basit tekrar temizliği
    subset = [c for c in ["name", "city", "district", "lat", "lon"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")

    debug["rows_after_normalize"] = len(df)

    return DataLoadResult(
        df=df,
        source_label=f"Türkiye Geneli CKAN Taraması · {len(df)} kayıt",
        mode="Türkiye Geneli CKAN Taraması",
        is_demo=False,
        errors=errors,
        debug=debug,
    )


def apply_strict_mode(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "has_valid_coordinate" in out.columns:
        out = out[out["has_valid_coordinate"] == True]

    if "source_portal" in out.columns:
        out = out[
            out["source_portal"].notna()
            & (out["source_portal"].astype(str).str.strip() != "")
        ]

    return out.reset_index(drop=True)


def load_shelter_dataset(
    mode: str,
    strict_mode: bool = False,
    demo_path: str = DEMO_CSV_PATH,
) -> DataLoadResult:
    """
    Ana veri yükleme fonksiyonu.

    Önemli:
    - Demo seçiliyse demo yükler.
    - CKAN seçiliyse CKAN yükler.
    - CKAN hata verirse demo CSV'ye otomatik dönmez.
    """

    normalized_mode = str(mode or "").strip().lower()

    if "ckan" in normalized_mode or "türkiye" in normalized_mode or "turkiye" in normalized_mode:
        result = load_ckan_data()

        if strict_mode and not result.df.empty:
            result.df = apply_strict_mode(result.df)

        return result

    try:
        df = load_demo_data(demo_path)

        if strict_mode and not df.empty:
            df = apply_strict_mode(df)

        return DataLoadResult(
            df=df,
            source_label=f"Stabil Demo CSV · {demo_path}",
            mode="Stabil Demo CSV",
            is_demo=True,
            errors=[],
            debug={"demo_path": demo_path, "rows": len(df)},
        )

    except Exception as e:
        return DataLoadResult(
            df=pd.DataFrame(),
            source_label=f"Stabil Demo CSV · okunamadı",
            mode="Stabil Demo CSV",
            is_demo=True,
            errors=[str(e)],
            debug={"demo_path": demo_path},
        )


# Eski importları kırmamak için alias fonksiyonlar
def load_data(
    mode: str = "Stabil Demo CSV",
    strict_mode: bool = False,
) -> DataLoadResult:
    return load_shelter_dataset(mode=mode, strict_mode=strict_mode)


def get_data(
    mode: str = "Stabil Demo CSV",
    strict_mode: bool = False,
) -> DataLoadResult:
    return load_shelter_dataset(mode=mode, strict_mode=strict_mode)
