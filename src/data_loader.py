import pandas as pd
import requests
import streamlit as st


LOCAL_FILE = "data/kocaeli_shelters.csv"


CKAN_SOURCES = {
    "Kocaeli Açık Veri": {
        "base": "https://veri.kocaeli.bel.tr",
        "query": "hayvan toplama merkezi",
    },
    "Ordu Açık Veri": {
        "base": "https://acikveri.ordu.bel.tr",
        "query": "hayvan bakımevi",
    },
    "B40 İstanbul": {
        "base": "https://opendata.b40cities.org",
        "query": "hayvan bakımevi",
    },
}


def safe_get_json(url, params=None, timeout=20):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def infer_format(resource):
    fmt = str(resource.get("format", "") or "").lower().strip()
    url = str(resource.get("url", "") or "").lower()

    if fmt:
        return fmt

    if url.endswith(".csv"):
        return "csv"
    if url.endswith(".xlsx"):
        return "xlsx"
    if url.endswith(".xls"):
        return "xls"
    if url.endswith(".json"):
        return "json"
    if url.endswith(".ods"):
        return "ods"

    return ""


@st.cache_data(ttl=3600)
def load_local_data(path=LOCAL_FILE):
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def search_ckan_resources(base_url, query):
    search_url = f"{base_url.rstrip('/')}/api/3/action/package_search"

    data = safe_get_json(
        search_url,
        params={
            "q": query,
            "rows": 20,
        },
    )

    results = data.get("result", {}).get("results", [])
    resources = []

    allowed_formats = {"csv", "xlsx", "xls", "json", "ods"}

    for package in results:
        package_title = package.get("title", "Veri Paketi")

        for res in package.get("resources", []):
            url = res.get("url", "")
            fmt = infer_format(res)
            name = res.get("name", package_title)

            if url and fmt in allowed_formats:
                resources.append(
                    {
                        "package": package_title,
                        "name": name,
                        "format": fmt,
                        "url": url,
                        "resource_id": res.get("id", ""),
                    }
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
                        props = feature.get("properties", {})
                        geometry = feature.get("geometry", {})

                        if geometry.get("type") == "Point":
                            coords = geometry.get("coordinates", [None, None])
                            props["lon"] = coords[0]
                            props["lat"] = coords[1]

                        rows.append(props)

                    return pd.DataFrame(rows)

                return pd.DataFrame([data])

    return pd.DataFrame()
