import pandas as pd
import requests
import streamlit as st


LOCAL_FILE = "data/kocaeli_shelters.csv"


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


def make_ckan_api_url(base_url):
    base = base_url.rstrip("/")

    if base.endswith("/api/3"):
        return f"{base}/action/package_search"

    if base.endswith("/api/3/action"):
        return f"{base}/package_search"

    return f"{base}/api/3/action/package_search"


def classify_resource(resource):
    """
    CKAN resource sınıflandırması.

    shelter_facility:
        Barınak, bakımevi, geçici hayvan bakım merkezi gibi envanter olma ihtimali yüksek kaynaklar.

    operation_stats:
        İşlem sayısı, istatistik, denetim, evcil hayvan varlığı gibi operasyonel/istatistiksel kaynaklar.

    general_animal:
        Hayvan/veteriner konulu ama barınak envanteri olduğu net olmayan kaynaklar.

    irrelevant:
        İlgisiz kaynaklar.
    """

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
        "hayvan bakımevi",
        "hayvan bakimevi",
        "geçici hayvan bakım merkezi",
        "gecici hayvan bakim merkezi",
        "geçici hayvan bakımevi",
        "gecici hayvan bakimevi",
        "hayvan bakım merkezi",
        "hayvan bakim merkezi",
        "hayvan toplama merkezi",
        "toplama merkezi",
        "sahipsiz hayvan rehabilitasyon",
        "rehabilitasyon merkezi",
        "barınak",
        "barinak",
        "bakımevi",
        "bakimevi",
    ]

    operation_keywords = [
        "işlem sayıları",
        "islem sayilari",
        "işlem_sayıları",
        "islem_sayilari",
        "işlemleri",
        "islemleri",
        "istatistik",
        "istatistikleri",
        "yıllara göre",
        "yillara gore",
        "denetim",
        "hanelerde",
        "evcil hayvan",
        "evcil hayvan varlığı",
        "evcil hayvan varligi",
        "evcil hayvan türleri",
        "evcil hayvan turleri",
        "vektör",
        "vektor",
        "mücadele",
        "mucadele",
        "sağlık kurum",
        "saglik kurum",
        "kuruluşlarına ilişkin",
        "kuruluslarina iliskin",
        "vdym",
        "sayısı",
        "sayisi",
    ]

    general_animal_keywords = [
        "hayvan",
        "veteriner",
        "sahipsiz",
        "kısırlaştırma",
        "kisirlastirma",
        "rehabilitasyon",
    ]

    if any(k in text for k in shelter_keywords):
        if any(k in text for k in operation_keywords):
            return "operation_stats"
        return "shelter_facility"

    if any(k in text for k in operation_keywords):
        return "operation_stats"

    if any(k in text for k in general_animal_keywords):
        return "general_animal"

    return "irrelevant"


def resource_relevance_score(resource):
    category = classify_resource(resource)

    score_map = {
        "shelter_facility": 100,
        "operation_stats": 45,
        "general_animal": 25,
        "irrelevant": 0,
    }

    return score_map.get(category, 0)


def is_relevant_resource(resource):
    return classify_resource(resource) in [
        "shelter_facility",
        "operation_stats",
        "general_animal",
    ]


@st.cache_data(ttl=3600)
def load_local_data(path=LOCAL_FILE):
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def search_ckan_resources(base_url, query, rows=50, deep_queries=None):
    search_url = make_ckan_api_url(base_url)

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

                item = {
                    "source_portal": "",
                    "source_base": base_url,
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

                item["resource_category"] = classify_resource(item)
                item["relevance_score"] = resource_relevance_score(item)

                resources.append(item)

    resources = [r for r in resources if is_relevant_resource(r)]

    resources = sorted(
        resources,
        key=lambda r: (
            int(r.get("relevance_score", 0)),
            str(r.get("resource_last_modified", "")),
            str(r.get("resource_revision_timestamp", "")),
            str(r.get("package_modified", "")),
        ),
        reverse=True,
    )

    return resources


@st.cache_data(ttl=3600)
def search_turkiye_ckan_resources(rows_per_query=50):
    all_resources = []
    seen_urls = set()

    for source_name, source in TURKIYE_CKAN_SOURCES.items():
        resources = search_ckan_resources(
            base_url=source["base"],
            query=source["query"],
            rows=rows_per_query,
            deep_queries=source.get("deep_queries", COMMON_DEEP_QUERIES),
        )

        for r in resources:
            url = r.get("url", "")

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            r["source_portal"] = source_name
            r["source_base"] = source["base"]
            r["resource_category"] = classify_resource(r)
            r["relevance_score"] = resource_relevance_score(r)
            all_resources.append(r)

    all_resources = sorted(
        all_resources,
        key=lambda r: (
            int(r.get("relevance_score", 0)),
            str(r.get("source_portal", "")),
            str(r.get("resource_last_modified", "")),
            str(r.get("package_modified", "")),
        ),
        reverse=True,
    )

    return all_resources


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


def load_resource_with_metadata(resource):
    try:
        df = load_resource(resource)
    except Exception:
        return pd.DataFrame()

    if df.empty:
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


def load_multiple_resources(
    resources,
    max_resources=20,
    allowed_categories=None,
):
    """
    Çoklu resource yükler.

    allowed_categories verilirse sadece o kategorideki kaynakları içeri alır.
    Ana dashboard için önerilen:
        allowed_categories=["shelter_facility"]
    """

    if allowed_categories is not None:
        resources = [
            r for r in resources
            if r.get("resource_category") in allowed_categories
        ]

    frames = []
    loaded_resources = []
    failed_resources = []

    for resource in resources[:max_resources]:
        df = load_resource_with_metadata(resource)

        if df.empty:
            failed_resources.append(resource)
            continue

        df["resource_category"] = resource.get("resource_category", "")
        df["relevance_score"] = resource.get("relevance_score", 0)

        frames.append(df)
        loaded_resources.append(resource)

    if not frames:
        return pd.DataFrame(), loaded_resources, failed_resources

    combined = pd.concat(frames, ignore_index=True, sort=False)

    return combined, loaded_resources, failed_resources
