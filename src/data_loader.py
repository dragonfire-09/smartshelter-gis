import pandas as pd
import requests
import streamlit as st


LOCAL_FILE = "data/kocaeli_shelters.csv"


CKAN_SOURCES = {
    "Kocaeli Açık Veri": {
        "base": "https://veri.kocaeli.bel.tr",
        "query": "hayvan toplama merkezi",
        "deep_queries": [
            "hayvan",
            "barınak",
            "barinak",
            "bakımevi",
            "bakimevi",
            "toplama merkezi",
            "sokak hayvan",
            "sokak hayvanları",
            "veteriner",
            "kısırlaştırma",
            "kisirlastirma",
            "rehabilitasyon",
            "geçici bakımevi",
            "gecici bakimevi",
        ],
    },
    "Ordu Açık Veri": {
        "base": "https://acikveri.ordu.bel.tr",
        "query": "hayvan bakımevi",
        "deep_queries": [
            "hayvan",
            "barınak",
            "barinak",
            "bakımevi",
            "bakimevi",
            "sokak hayvan",
            "sokak hayvanları",
            "veteriner",
            "rehabilitasyon",
            "kısırlaştırma",
            "kisirlastirma",
        ],
    },
    "B40 İstanbul": {
        "base": "https://opendata.b40cities.org",
        "query": "hayvan bakımevi",
        "deep_queries": [
            "hayvan",
            "barınak",
            "barinak",
            "bakımevi",
            "bakimevi",
            "sokak hayvan",
            "sokak hayvanları",
            "veteriner",
            "rehabilitasyon",
            "kısırlaştırma",
            "kisirlastirma",
        ],
    },
}


def safe_get_json(url, params=None, timeout=25):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def infer_format(resource):
    fmt = str(resource.get("format", "") or "").lower().strip()
    url = str(resource.get("url", "") or "").lower()

    if fmt:
        fmt = fmt.replace(".", "").strip()

    if fmt in ["csv", "xlsx", "xls", "json", "ods"]:
        return fmt

    if url.endswith(".csv"):
        return "csv"
    if url.endswith(".xlsx"):
        return "xlsx"
    if url.endswith(".xls"):
        return "xls"
    if url.endswith(".json") or "format=json" in url:
        return "json"
    if url.endswith(".ods"):
        return "ods"

    return ""


@st.cache_data(ttl=3600)
def load_local_data(path=LOCAL_FILE):
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def search_ckan_resources(base_url, query, rows=50, deep_queries=None):
    """
    CKAN package_search ile kaynak arar.

    deep_queries verilirse sadece ana query değil, eski veya farklı isimlendirilmiş
    resource'ları bulmak için ek anahtar kelimelerle de arama yapar.
    """
    search_url = f"{base_url.rstrip('/')}/api/3/action/package_search"

    queries = [query]

    if deep_queries:
        queries.extend(deep_queries)

    queries = list(dict.fromkeys([q for q in queries if q]))

    resources = []
    allowed_formats = {"csv", "xlsx", "xls", "json", "ods"}

    seen_urls = set()

    for q in queries:
        try:
            data = safe_get_json(
                search_url,
                params={
                    "q": q,
                    "rows": rows,
                },
            )
        except Exception:
            continue

        results = data.get("result", {}).get("results", [])

        for package in results:
            package_title = package.get("title", "Veri Paketi")
            package_name = package.get("name", "")
            package_notes = package.get("notes", "")
            package_created = package.get("metadata_created", "")
            package_modified = package.get("metadata_modified", "")

            for res in package.get("resources", []):
                url = res.get("url", "")
                fmt = infer_format(res)
                name = res.get("name", package_title)

                if not url:
                    continue

                if url in seen_urls:
                    continue

                if fmt not in allowed_formats:
                    continue

                seen_urls.add(url)

                resources.append(
                    {
                        "package": package_title,
                        "package_name": package_name,
                        "package_notes": package_notes,
                        "name": name,
                        "format": fmt,
                        "url": url,
                        "resource_id": res.get("id", ""),
                        "package_created": package_created,
                        "package_modified": package_modified,
                        "resource_created": res.get("created", ""),
                        "resource_last_modified": res.get("last_modified", ""),
                        "resource_revision_timestamp": res.get("revision_timestamp", ""),
                        "matched_query": q,
                    }
                )

    resources = sorted(
        resources,
        key=lambda r: (
            str(r.get("resource_last_modified", "")),
            str(r.get("resource_revision_timestamp", "")),
            str(r.get("package_modified", "")),
        ),
        reverse=True,
    )

    return resources


@st.cache_data(ttl=3600)
def load_resource(resource):
    fmt = resource["format"]
    url = resource["url"]

    if fmt == "csv":
        return pd.read_csv(url)

    if fmt in ["xlsx", "xls"]:
        return pd.read_excel(url)

    if fmt == "ods":
        return pd.read_excel(url, engine="odf")

    if fmt == "json":
        try:
            return pd.read_json(url)
        except Exception:
            data = safe_get_json(url)

            if isinstance(data, list):
                return pd.DataFrame(data)

            if isinstance(data, dict):
                if "records" in data and isinstance(data["records"], list):
                    return pd.DataFrame(data["records"])

                if (
                    "result" in data
                    and isinstance(data["result"], dict)
                    and "records" in data["result"]
                ):
                    return pd.DataFrame(data["result"]["records"])

                if "features" in data and isinstance(data["features"], list):
                    rows = []

                    for feature in data["features"]:
                        props = feature.get("properties", {}) or {}
                        geometry = feature.get("geometry", {}) or {}

                        if geometry.get("type") == "Point":
                            coords = geometry.get("coordinates", [None, None])
                            props["lon"] = coords[0]
                            props["lat"] = coords[1]

                        rows.append(props)

                    return pd.DataFrame(rows)

                return pd.DataFrame([data])

    return pd.DataFrame()
