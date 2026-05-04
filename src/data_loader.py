import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st


LOCAL_FILE = "data/kocaeli_shelters.csv"

HTTP_TIMEOUT = 8
CKAN_SEARCH_WORKERS = 12
RESOURCE_LOAD_WORKERS = 6


COMMON_DEEP_QUERIES = [
    "hayvan",
    "barınak",
    "barinak",
    "bakımevi",
    "bakimevi",
    "hayvan bakımevi",
    "hayvan bakimevi",
    "geçici hayvan bakımevi",
    "gecici hayvan bakimevi",
    "geçici hayvan bakım merkezi",
    "gecici hayvan bakim merkezi",
    "toplama merkezi",
    "hayvan toplama merkezi",
    "sokak hayvan",
    "sokak hayvanları",
    "sahipsiz hayvan",
    "veteriner",
    "kısırlaştırma",
    "kisirlastirma",
    "rehabilitasyon",
]


CKAN_SOURCES = {
    "Kocaeli Açık Veri": {
        "base": "https://veri.kocaeli.bel.tr",
        "query": "hayvan toplama merkezi",
        "deep_queries": COMMON_DEEP_QUERIES,
    },
    "Ordu Açık Veri": {
        "base": "https://acikveri.ordu.bel.tr",
        "query": "hayvan bakımevi",
        "deep_queries": COMMON_DEEP_QUERIES,
    },
    "B40 İstanbul": {
        "base": "https://opendata.b40cities.org",
        "query": "hayvan bakımevi",
        "deep_queries": COMMON_DEEP_QUERIES,
    },
}


TURKIYE_CKAN_SOURCES = {
    "İBB Açık Veri": {
        "base": "https://data.ibb.gov.tr",
        "query": "hayvan",
        "deep_queries": COMMON_DEEP_QUERIES,
    },
    "İzmir Açık Veri": {
        "base": "https://acikveri.bizizmir.com",
        "query": "hayvan",
        "deep_queries": COMMON_DEEP_QUERIES,
    },
    "Konya Açık Veri": {
        "base": "https://acikveri.konya.bel.tr",
        "query": "hayvan",
        "deep_queries": COMMON_DEEP_QUERIES,
    },
    "Kocaeli Açık Veri": {
        "base": "https://veri.kocaeli.bel.tr",
        "query": "hayvan",
        "deep_queries": COMMON_DEEP_QUERIES,
    },
    "Ordu Açık Veri": {
        "base": "https://acikveri.ordu.bel.tr",
        "query": "hayvan",
        "deep_queries": COMMON_DEEP_QUERIES,
    },
    "Gaziantep Açık Veri": {
        "base": "https://acikveri.gaziantep.bel.tr",
        "query": "hayvan",
        "deep_queries": COMMON_DEEP_QUERIES,
    },
    "Kadıköy Açık Veri": {
        "base": "https://acikveri.kadikoy.bel.tr",
        "query": "hayvan",
        "deep_queries": COMMON_DEEP_QUERIES,
    },
    "Tuzla Açık Veri": {
        "base": "https://veri.tuzla.bel.tr",
        "query": "hayvan",
        "deep_queries": COMMON_DEEP_QUERIES,
    },
    "B40 Açık Veri": {
        "base": "https://opendata.b40cities.org",
        "query": "hayvan",
        "deep_queries": COMMON_DEEP_QUERIES,
    },
}


# ---------------------------------------------------------
# HTTP Helpers
# ---------------------------------------------------------
def safe_get_json(url, params=None, timeout=HTTP_TIMEOUT):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def safe_get_bytes(url, timeout=HTTP_TIMEOUT * 2):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


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


def make_ckan_api_url(base_url):
    base = base_url.rstrip("/")

    if base.endswith("/api/3"):
        return f"{base}/action/package_search"

    if base.endswith("/api/3/action"):
        return f"{base}/package_search"

    return f"{base}/api/3/action/package_search"


# ---------------------------------------------------------
# Classification
# ---------------------------------------------------------
def classify_resource(resource):
    text = " ".join(
        [
            str(resource.get("source_portal", "")),
            str(resource.get("package", "")),
            str(resource.get("name", "")),
            str(resource.get("package_notes", "")),
            str(resource.get("matched_query", "")),
        ]
    ).lower()

    shelter_keywords = [
        "hayvan bakımevi", "hayvan bakimevi",
        "geçici hayvan bakım merkezi", "gecici hayvan bakim merkezi",
        "geçici hayvan bakımevi", "gecici hayvan bakimevi",
        "hayvan bakım merkezi", "hayvan bakim merkezi",
        "hayvan toplama merkezi", "toplama merkezi",
        "sahipsiz hayvan rehabilitasyon", "rehabilitasyon merkezi",
        "barınak", "barinak", "bakımevi", "bakimevi",
        "animal shelter", "shelter",
    ]

    operation_keywords = [
        "işlem sayıları", "islem sayilari",
        "işlemleri", "islemleri",
        "istatistik", "istatistikleri",
        "yıllara göre", "yillara gore",
        "denetim", "hanelerde",
        "evcil hayvan", "evcil hayvan varlığı", "evcil hayvan varligi",
        "evcil hayvan türleri", "evcil hayvan turleri",
        "vektör", "vektor", "mücadele", "mucadele",
        "sağlık kurum", "saglik kurum",
        "vdym", "sayısı", "sayisi",
        "number and capacities", "by years", "years",
    ]

    general_keywords = [
        "hayvan", "veteriner", "sahipsiz",
        "kısırlaştırma", "kisirlastirma", "rehabilitasyon",
    ]

    if any(k in text for k in shelter_keywords):
        if any(k in text for k in operation_keywords):
            return "operation_stats"
        return "shelter_facility"

    if any(k in text for k in operation_keywords):
        return "operation_stats"

    if any(k in text for k in general_keywords):
        return "general_animal"

    return "irrelevant"


def resource_relevance_score(resource):
    score_map = {
        "shelter_facility": 100,
        "operation_stats": 60,
        "general_animal": 25,
        "irrelevant": 0,
    }
    return score_map.get(classify_resource(resource), 0)


def is_relevant_resource(resource):
    return classify_resource(resource) in [
        "shelter_facility",
        "operation_stats",
        "general_animal",
    ]


# ---------------------------------------------------------
# Local
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_local_data(path=LOCAL_FILE):
    return pd.read_csv(path)


# ---------------------------------------------------------
# Single CKAN search query
# ---------------------------------------------------------
def _search_single_query(search_url, query, rows):
    try:
        data = safe_get_json(
            search_url,
            params={"q": query, "rows": rows},
            timeout=HTTP_TIMEOUT,
        )
    except Exception:
        return []

    out = []
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

            if not url or fmt not in {"csv", "xlsx", "xls", "json", "ods"}:
                continue

            item = {
                "source_portal": "",
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
                "matched_query": query,
            }

            item["resource_category"] = classify_resource(item)
            item["relevance_score"] = resource_relevance_score(item)

            out.append(item)

    return out


@st.cache_data(ttl=3600, show_spinner=False)
def search_ckan_resources(base_url, query, rows=50, deep_queries=None):
    """Tek bir CKAN portalında paralel sorgular."""
    search_url = make_ckan_api_url(base_url)

    queries = [query]
    if deep_queries:
        queries.extend(deep_queries)
    queries = list(dict.fromkeys([q for q in queries if q]))

    all_items = []
    seen_urls = set()

    with ThreadPoolExecutor(max_workers=CKAN_SEARCH_WORKERS) as executor:
        futures = {
            executor.submit(_search_single_query, search_url, q, rows): q
            for q in queries
        }

        for future in as_completed(futures):
            try:
                items = future.result(timeout=HTTP_TIMEOUT * 2)
            except Exception:
                continue

            for item in items:
                url = item.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                item["source_base"] = base_url
                all_items.append(item)

    all_items = [r for r in all_items if is_relevant_resource(r)]

    all_items.sort(
        key=lambda r: (
            int(r.get("relevance_score", 0)),
            str(r.get("resource_last_modified", "")),
            str(r.get("resource_revision_timestamp", "")),
            str(r.get("package_modified", "")),
        ),
        reverse=True,
    )

    return all_items


# ---------------------------------------------------------
# Multi-portal CKAN search
# ---------------------------------------------------------
def _search_one_portal(source_name, source_config, rows_per_query):
    try:
        resources = search_ckan_resources(
            base_url=source_config["base"],
            query=source_config["query"],
            rows=rows_per_query,
            deep_queries=source_config.get("deep_queries", COMMON_DEEP_QUERIES),
        )
    except Exception:
        return source_name, []

    for r in resources:
        r["source_portal"] = source_name
        r["source_base"] = source_config["base"]
        r["resource_category"] = classify_resource(r)
        r["relevance_score"] = resource_relevance_score(r)

    return source_name, resources


@st.cache_data(ttl=3600, show_spinner=False)
def search_turkiye_ckan_resources(rows_per_query=50):
    """Türkiye geneli paralel CKAN taraması (cache'li)."""
    return _search_turkiye_ckan_internal(rows_per_query=rows_per_query)


def search_turkiye_ckan_resources_with_progress(
    rows_per_query=50,
    on_portal_done=None,
):
    """Callback'li paralel tarama. on_portal_done(portal_name, idx, total) çağrılır."""
    return _search_turkiye_ckan_internal(
        rows_per_query=rows_per_query,
        on_portal_done=on_portal_done,
    )


def _search_turkiye_ckan_internal(rows_per_query=50, on_portal_done=None):
    all_resources = []
    seen_urls = set()
    portal_items = list(TURKIYE_CKAN_SOURCES.items())
    total = len(portal_items)

    with ThreadPoolExecutor(max_workers=CKAN_SEARCH_WORKERS) as executor:
        futures = {
            executor.submit(
                _search_one_portal,
                source_name,
                source_config,
                rows_per_query,
            ): source_name
            for source_name, source_config in portal_items
        }

        done = 0
        for future in as_completed(futures):
            done += 1
            portal_name = futures[future]

            try:
                _, resources = future.result(timeout=120)
            except Exception:
                resources = []

            if on_portal_done:
                try:
                    on_portal_done(portal_name, done, total)
                except Exception:
                    pass

            for r in resources:
                url = r.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                all_resources.append(r)

    all_resources.sort(
        key=lambda r: (
            int(r.get("relevance_score", 0)),
            str(r.get("source_portal", "")),
            str(r.get("resource_last_modified", "")),
            str(r.get("package_modified", "")),
        ),
        reverse=True,
    )

    return all_resources


# ---------------------------------------------------------
# Resource loaders
# ---------------------------------------------------------
def _read_csv_smart(content_bytes):
    for encoding in ["utf-8", "utf-8-sig", "cp1254", "iso-8859-9", "latin-1"]:
        try:
            return pd.read_csv(io.BytesIO(content_bytes), encoding=encoding)
        except Exception:
            continue

    try:
        return pd.read_csv(io.BytesIO(content_bytes), encoding="utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame()


def _read_excel_smart(content_bytes, engine=None):
    try:
        if engine:
            return pd.read_excel(io.BytesIO(content_bytes), engine=engine)
        return pd.read_excel(io.BytesIO(content_bytes))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_resource(resource):
    fmt = resource.get("format", "")
    url = resource.get("url", "")

    if not url:
        return pd.DataFrame()

    try:
        content = safe_get_bytes(url)
    except Exception:
        return pd.DataFrame()

    if fmt == "csv":
        return _read_csv_smart(content)

    if fmt in ["xlsx", "xls"]:
        return _read_excel_smart(content)

    if fmt == "ods":
        return _read_excel_smart(content, engine="odf")

    if fmt == "json":
        try:
            return pd.read_json(io.BytesIO(content))
        except Exception:
            pass

        try:
            import json
            data = json.loads(content.decode("utf-8", errors="ignore"))
        except Exception:
            return pd.DataFrame()

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


def load_resource_with_metadata(resource):
    try:
        df = load_resource(resource)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    df["source_portal"] = resource.get("source_portal", "")
    df["source_resource"] = (
        f"{resource.get('package', '')} | {resource.get('name', '')}"
    )
    df["source_url"] = resource.get("url", "")
    df["source_format"] = resource.get("format", "")
    df["source_matched_query"] = resource.get("matched_query", "")
    df["source_package_modified"] = resource.get("package_modified", "")
    df["source_resource_last_modified"] = resource.get("resource_last_modified", "")
    df["resource_category"] = resource.get("resource_category", "")
    df["relevance_score"] = resource.get("relevance_score", 0)

    return df


def load_multiple_resources(resources, max_resources=20, allowed_categories=None, on_resource_done=None):
    if allowed_categories is not None:
        resources = [
            r for r in resources
            if r.get("resource_category") in allowed_categories
        ]

    targets = list(resources[:max_resources])

    frames = []
    loaded_resources = []
    failed_resources = []

    if not targets:
        return pd.DataFrame(), loaded_resources, failed_resources

    total = len(targets)

    with ThreadPoolExecutor(max_workers=RESOURCE_LOAD_WORKERS) as executor:
        futures = {
            executor.submit(load_resource_with_metadata, resource): resource
            for resource in targets
        }

        done = 0
        for future in as_completed(futures):
            done += 1
            resource = futures[future]

            try:
                df = future.result(timeout=60)
            except Exception:
                df = None

            if on_resource_done:
                try:
                    on_resource_done(resource, done, total)
                except Exception:
                    pass

            if df is None or df.empty:
                failed_resources.append(resource)
                continue

            frames.append(df)
            loaded_resources.append(resource)

    if not frames:
        return pd.DataFrame(), loaded_resources, failed_resources

    combined = pd.concat(frames, ignore_index=True, sort=False)

    return combined, loaded_resources, failed_resources
