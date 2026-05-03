from io import BytesIO
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.ai_insights import (
    calculate_data_quality_score,
    detect_anomalies,
    generate_executive_summary,
    generate_risk_explanations,
    simulate_interventions,
)
from src.charts import (
    build_district_summary,
    chart_district_avg_risk,
    chart_history_metric,
    chart_history_trend,
    chart_occupancy_rate,
    chart_record_delta,
    chart_risk_score,
)
from src.data_loader import (
    CKAN_SOURCES,
    LOCAL_FILE,
    TURKIYE_CKAN_SOURCES,
    load_local_data,
    load_multiple_resources,
    load_resource,
    search_ckan_resources,
    search_turkiye_ckan_resources,
)
from src.history import (
    append_snapshot,
    build_history_summary,
    compare_snapshot_dates,
    compare_summary,
    get_available_snapshot_dates,
)
from src.map import create_shelter_map
from src.normalize import normalize_columns
from src.risk import calculate_risk, create_action_recommendations


# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="SmartShelter GIS",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

HISTORY_FILE = Path("data/history/shelter_history.csv")


# ---------------------------------------------------------
# Cache Wrappers
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_load_local_data(path):
    return load_local_data(path)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_search_ckan_resources(base, query, rows, deep_queries_tuple):
    deep_queries = list(deep_queries_tuple) if deep_queries_tuple else None

    return search_ckan_resources(
        base,
        query,
        rows=rows,
        deep_queries=deep_queries,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def cached_search_turkiye_ckan_resources(rows_per_query):
    return search_turkiye_ckan_resources(rows_per_query=rows_per_query)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_load_resource(resource_tuple):
    """resource_tuple: tuple(sorted(dict.items())) - hashlenebilir."""
    resource = dict(resource_tuple)
    return load_resource(resource)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_load_multiple_resources(resources_tuple, max_resources):
    """resources_tuple: tuple of tuple(sorted(dict.items())) - hashlenebilir."""
    resources = [dict(items) for items in resources_tuple]

    return load_multiple_resources(
        resources,
        max_resources=max_resources,
    )


def to_resource_tuple(resource: dict) -> tuple:
    safe_items = []
    for k, v in resource.items():
        if isinstance(v, (list, dict, set)):
            safe_items.append((k, str(v)))
        else:
            safe_items.append((k, v))

    return tuple(sorted(safe_items, key=lambda kv: kv[0]))


# ---------------------------------------------------------
# CSS / UI
# ---------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
            border-right: 1px solid #e5e7eb;
        }

        div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.92);
            border: 1px solid #e5e7eb;
            padding: 16px;
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
        }

        .hero {
            padding: 26px 30px;
            border-radius: 28px;
            background:
                radial-gradient(circle at top left, rgba(59,130,246,0.28), transparent 30%),
                radial-gradient(circle at bottom right, rgba(16,185,129,0.20), transparent 30%),
                linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #334155 100%);
            color: white;
            box-shadow: 0 18px 45px rgba(15,23,42,0.22);
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: 2.25rem;
            line-height: 1.15;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.04em;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            color: #dbeafe;
            margin-top: 0.7rem;
            max-width: 850px;
        }

        .hero-badges {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 18px;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.18);
            color: #f8fafc;
            font-size: 0.86rem;
            backdrop-filter: blur(8px);
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }

        .section-caption {
            color: #64748b;
            font-size: 0.92rem;
            margin-bottom: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🐾 SmartShelter GIS</div>
            <div class="hero-subtitle">
                Açık veri tabanlı hayvan bakımevi izleme, risk önceliklendirme,
                veri kalite analizi ve GIS karar destek paneli.
            </div>
            <div class="hero-badges">
                <span class="pill">📍 GIS Harita</span>
                <span class="pill">⚠️ Risk Skoru</span>
                <span class="pill">🧠 Kural Tabanlı AI Analiz</span>
                <span class="pill">📊 CKAN Açık Veri</span>
                <span class="pill">🕒 Tarihsel Snapshot</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def render_methodology_note():
    st.info(
        """
        Bu uygulama resmi bir denetim sistemi değildir. Açık veri ve demo veriyle çalışan
        prototip bir karar destek ekranıdır.

        Sistem Türkiye geneli açık veri kaynaklarını tek havuzda toplar; ancak verileri
        risk analizi, kapasite verisi, tesis/konum verisi ve operasyonel istatistikler
        olarak ayırır. Böylece yalnızca kapasite istatistiği veya işlem sayısı içeren
        kaynaklar risk hesabına karıştırılmaz.

        AI Analiz bölümü harici AI API kullanmadan, şeffaf ve kural tabanlı çalışır.
        """
    )


def safe_plotly(fig_func, data, empty_message="Gösterilecek veri bulunamadı.", **kwargs):
    try:
        if data is None or len(data) == 0:
            st.warning(empty_message)
            return

        fig = fig_func(data, **kwargs) if kwargs else fig_func(data)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning("Grafik oluşturulamadı. Veri yapısı bu grafik için uygun olmayabilir.")
        with st.expander("Teknik hata detayı"):
            st.exception(e)


def to_excel_bytes(df: pd.DataFrame, district_summary: pd.DataFrame) -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Kayitlar")
        district_summary.to_excel(writer, index=False, sheet_name="Ilce_Ozeti")

    return output.getvalue()


# ---------------------------------------------------------
# Data Safety Helpers
# ---------------------------------------------------------
def as_bool_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)

    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "evet", "var", "available"])
    )


def ensure_app_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    string_defaults = {
        "name": "Bilinmeyen Kayıt",
        "city": "Bilinmiyor",
        "district": "Bilinmiyor",
        "source_portal": "",
        "source_resource": "",
        "source_url": "",
        "resource_category": "",
        "data_scope": "",
        "risk_level": "Veri yetersiz",
        "recommended_action": "Veri kalitesi artırılmalı ve eksik alanlar tamamlanmalıdır.",
        "risk_explanation": "",
        "data_quality_level": "Bilinmiyor",
        "data_quality_note": "",
        "analytics_exclusion_reason": "",
    }

    numeric_defaults = {
        "capacity": 0,
        "occupancy": 0,
        "vet_count": 0,
        "sterilization_count": 0,
        "adoption_count": 0,
        "occupancy_rate": pd.NA,
        "animals_per_vet": pd.NA,
        "risk_score": pd.NA,
        "data_quality_score": pd.NA,
        "latitude": pd.NA,
        "longitude": pd.NA,
        "relevance_score": 0,
    }

    for col, default in string_defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default

        if col not in ["latitude", "longitude"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "capacity",
        "occupancy",
        "vet_count",
        "sterilization_count",
        "adoption_count",
        "relevance_score",
    ]:
        df[col] = df[col].fillna(0)

    if "capacity_available" not in df.columns:
        df["capacity_available"] = df["capacity"].fillna(0) > 0
    else:
        df["capacity_available"] = as_bool_series(df["capacity_available"])

    if "occupancy_available" not in df.columns:
        df["occupancy_available"] = df["occupancy"].fillna(0) > 0
    else:
        df["occupancy_available"] = as_bool_series(df["occupancy_available"])

    if "vet_count_available" not in df.columns:
        df["vet_count_available"] = df["vet_count"].fillna(0) > 0
    else:
        df["vet_count_available"] = as_bool_series(df["vet_count_available"])

    if "coordinate_valid" not in df.columns:
        lat_ok = pd.to_numeric(df["latitude"], errors="coerce").between(-90, 90)
        lon_ok = pd.to_numeric(df["longitude"], errors="coerce").between(-180, 180)
        df["coordinate_valid"] = lat_ok & lon_ok
    else:
        df["coordinate_valid"] = as_bool_series(df["coordinate_valid"])

    if "analytics_eligible" not in df.columns:
        df["analytics_eligible"] = True
    else:
        df["analytics_eligible"] = as_bool_series(df["analytics_eligible"])

    if "risk_eligible" not in df.columns:
        df["risk_eligible"] = df["capacity_available"] & df["occupancy_available"]
    else:
        df["risk_eligible"] = as_bool_series(df["risk_eligible"])

    if df["data_scope"].astype(str).eq("").all():
        df.loc[df["risk_eligible"], "data_scope"] = "risk_ready"
        df.loc[
            ~df["risk_eligible"] & df["capacity_available"],
            "data_scope",
        ] = "capacity_only"
        df.loc[
            ~df["risk_eligible"] & ~df["capacity_available"] & df["coordinate_valid"],
            "data_scope",
        ] = "location_only"
        df.loc[df["data_scope"].astype(str).eq(""), "data_scope"] = "unknown"

    return df


def load_history_file() -> pd.DataFrame:
    if HISTORY_FILE.exists():
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def fallback_local(reason: str):
    st.warning(reason)

    raw = cached_load_local_data(LOCAL_FILE).copy()

    return raw, "Fallback Demo CSV", "Lokal CSV"


# ---------------------------------------------------------
# Render Blocks
# ---------------------------------------------------------
def render_data_source_status(
    selected_source_name,
    selected_resource_label,
    deep_scan,
    candidate_resources_df,
    loaded_resources_info,
    df_all_loaded,
    excluded_df,
    failed_resource_count,
    mode,
):
    with st.expander("ℹ️ Prototip ve Veri Kaynağı Bilgisi", expanded=False):
        render_methodology_note()

        c1, c2 = st.columns([1.1, 1.9])

        with c1:
            st.write(f"**Aktif veri kaynağı:** {selected_source_name}")
            st.write(f"**Resource:** {selected_resource_label}")
            st.write(f"**Derin CKAN taraması:** {'Açık' if deep_scan else 'Kapalı'}")
            st.write("**Tarihsel snapshot:** `data/history/shelter_history.csv`")

            if mode == "Türkiye Geneli CKAN Taraması":
                st.write(f"**Türkiye geneli CKAN kaynak sayısı:** {len(TURKIYE_CKAN_SOURCES)}")
                st.write(f"**Başarısız/boş resource:** {failed_resource_count}")

        with c2:
            k1, k2, k3, k4 = st.columns(4)

            k1.metric("Aday Resource", len(candidate_resources_df))
            k2.metric("İçeri Alınan", len(loaded_resources_info))
            k3.metric("Yüklenen Satır", len(df_all_loaded))
            k4.metric("Dışlanan Satır", len(excluded_df))

        if not excluded_df.empty:
            st.markdown("#### Dışlanan Kayıt Örnekleri")

            display_cols = [
                "name",
                "city",
                "district",
                "source_portal",
                "source_resource",
                "resource_category",
                "data_scope",
                "analytics_exclusion_reason",
            ]

            display_cols = [c for c in display_cols if c in excluded_df.columns]

            st.dataframe(
                excluded_df[display_cols].head(300),
                use_container_width=True,
                hide_index=True,
            )


def render_kpis(df: pd.DataFrame):
    total_records = len(df)

    if total_records == 0:
        st.warning("Filtreye uygun kayıt bulunamadı.")
        return

    capacity_available = (
        as_bool_series(df["capacity_available"])
        if "capacity_available" in df.columns
        else pd.Series([False] * len(df))
    )
    occupancy_available = (
        as_bool_series(df["occupancy_available"])
        if "occupancy_available" in df.columns
        else pd.Series([False] * len(df))
    )
    risk_eligible = (
        as_bool_series(df["risk_eligible"])
        if "risk_eligible" in df.columns
        else pd.Series([True] * len(df))
    )

    known_capacity = (
        int(df.loc[capacity_available, "capacity"].sum())
        if "capacity" in df.columns
        else 0
    )
    known_occupancy = (
        int(df.loc[occupancy_available, "occupancy"].sum())
        if "occupancy" in df.columns
        else 0
    )

    risk_df = df[risk_eligible].copy()

    avg_risk = (
        risk_df["risk_score"].mean()
        if len(risk_df) and "risk_score" in risk_df.columns
        else pd.NA
    )
    critical_count = (
        len(risk_df[risk_df["risk_level"].astype(str) == "Kritik"])
        if "risk_level" in risk_df.columns
        else 0
    )

    capacity_only_count = (
        len(df[df["data_scope"].astype(str).eq("capacity_only")])
        if "data_scope" in df.columns
        else 0
    )

    location_only_count = (
        len(df[df["data_scope"].astype(str).eq("location_only")])
        if "data_scope" in df.columns
        else 0
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Envanter Kaydı", total_records)
    c2.metric("Risk Hazır", len(risk_df))
    c3.metric("Kapasite Kaydı", capacity_only_count)
    c4.metric("Konum Kaydı", location_only_count)
    c5.metric("Bilinen Kapasite", known_capacity)
    c6.metric("Ortalama Risk", f"{avg_risk:.1f}" if pd.notna(avg_risk) else "Yok")

    c7, c8, c9 = st.columns(3)

    c7.metric("Bilinen Mevcut Hayvan", known_occupancy)
    c8.metric("Kritik Kayıt", critical_count)
    c9.metric(
        "Koordinatı Geçerli",
        int(as_bool_series(df["coordinate_valid"]).sum())
        if "coordinate_valid" in df.columns
        else 0,
    )

    if critical_count > 0:
        st.error(f"🔴 {critical_count} kayıt kritik risk seviyesinde.")
    else:
        st.success("🟢 Kritik seviyede kayıt bulunmuyor veya risk için yeterli veri yok.")


def render_record_detail(df: pd.DataFrame):
    section_header(
        "🏥 Kayıt Detayı",
        "Seçilen barınak/bakımevi veya veri kaydı için operasyonel özet.",
    )

    if df.empty:
        st.warning("Filtreye uygun kayıt bulunamadı.")
        return

    view = df.reset_index(drop=True).copy()

    view["record_label"] = (
        view["name"].astype(str)
        + " · "
        + view["city"].astype(str)
        + " / "
        + view["district"].astype(str)
        + " · #"
        + view.index.astype(str)
    )

    selected_idx = st.selectbox(
        "Kayıt seç",
        options=view.index.tolist(),
        format_func=lambda i: view.loc[i, "record_label"],
        key="record_detail_select",
    )

    item = view.loc[selected_idx]

    st.markdown("##### Temel Bilgiler")
    st.write(f"**Ad:** {item.get('name', '')}")
    st.write(f"**İl:** {item.get('city', '')}")
    st.write(f"**İlçe:** {item.get('district', '')}")
    st.write(f"**Veri Kapsamı:** `{item.get('data_scope', '')}`")

    if str(item.get("source_portal", "")).strip():
        st.write(f"**Kaynak Portal:** {item.get('source_portal', '')}")

    if str(item.get("source_resource", "")).strip():
        st.write(f"**Kaynak Resource:** {item.get('source_resource', '')}")

    st.markdown("##### Operasyonel Alanlar")

    if bool(item.get("capacity_available", False)):
        st.write(f"**Kapasite:** {int(item.get('capacity', 0))}")
    else:
        st.write("**Kapasite:** Veri yok")

    if bool(item.get("occupancy_available", False)):
        st.write(f"**Mevcut Hayvan:** {int(item.get('occupancy', 0))}")
    else:
        st.write("**Mevcut Hayvan:** Veri yok")

    if pd.notna(item.get("occupancy_rate", pd.NA)):
        st.write(f"**Doluluk Oranı:** %{item.get('occupancy_rate')}")
    else:
        st.write("**Doluluk Oranı:** Hesaplanamadı")

    if bool(item.get("vet_count_available", False)):
        st.write(f"**Veteriner Sayısı:** {int(item.get('vet_count', 0))}")
    else:
        st.write("**Veteriner Sayısı:** Veri yok")

    if pd.notna(item.get("animals_per_vet", pd.NA)):
        st.write(f"**Veteriner Başına Hayvan:** {float(item.get('animals_per_vet')):.1f}")

    st.markdown("##### Risk")

    risk_level = str(item.get("risk_level", "Veri yetersiz"))
    risk_score = item.get("risk_score", pd.NA)

    if pd.notna(risk_score):
        st.write(f"**Risk Skoru:** {risk_score}")
    else:
        st.write("**Risk Skoru:** Veri yetersiz")

    if risk_level == "Kritik":
        st.error("🔴 Kritik risk seviyesi")
    elif risk_level == "Orta":
        st.warning("🟠 Orta risk seviyesi")
    elif risk_level == "Düşük":
        st.success("🟢 Düşük risk seviyesi")
    else:
        st.info("ℹ️ Risk için yeterli veri yok")

    st.markdown("##### Önerilen Aksiyon")
    st.write(item.get("recommended_action", ""))

    if str(item.get("risk_explanation", "")).strip():
        st.markdown("##### AI Risk Açıklaması")
        st.info(item.get("risk_explanation", ""))

    if pd.notna(item.get("data_quality_score", pd.NA)):
        st.markdown("##### Veri Kalitesi")
        st.write(
            f"**{int(item.get('data_quality_score', 0))}/100** - "
            f"{item.get('data_quality_level', '')}"
        )

    if str(item.get("data_quality_note", "")).strip():
        st.warning(item.get("data_quality_note", ""))


def render_data_quality_summary(df: pd.DataFrame):
    section_header(
        "🧪 Veri Kalitesi Özeti",
        "Eksik koordinat, kapasite, mevcut hayvan ve düşük kalite kayıtların hızlı özeti.",
    )

    if df.empty:
        st.warning("Veri bulunamadı.")
        return

    coord_missing = (
        len(df[~as_bool_series(df["coordinate_valid"])])
        if "coordinate_valid" in df.columns
        else 0
    )
    low_quality = (
        len(df[df["data_quality_level"].astype(str) == "Düşük"])
        if "data_quality_level" in df.columns
        else 0
    )
    capacity_missing = (
        len(df[~as_bool_series(df["capacity_available"])])
        if "capacity_available" in df.columns
        else 0
    )
    occupancy_missing = (
        len(df[~as_bool_series(df["occupancy_available"])])
        if "occupancy_available" in df.columns
        else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Kayıt", len(df))
    c2.metric("Geçersiz Koordinat", coord_missing)
    c3.metric("Düşük Kalite", low_quality)
    c4.metric("Kapasite Eksik", capacity_missing)
    c5.metric("Mevcut Hayvan Eksik", occupancy_missing)


def render_history_analytics(history_df: pd.DataFrame, history_summary_df: pd.DataFrame):
    section_header(
        "🕒 Geçmiş Analitik ve Tarihsel Karşılaştırma",
        "Farklı snapshot tarihleri arasında kapasite, doluluk ve risk değişimini karşılaştırır.",
    )

    if history_df is None or history_df.empty:
        st.warning("Henüz tarihsel kayıt bulunmuyor.")
        return

    available_dates = get_available_snapshot_dates(history_df)

    if len(available_dates) == 0:
        st.warning("Henüz tarihsel kayıt bulunmuyor.")
        return

    if len(available_dates) == 1:
        st.info(
            f"Şu anda yalnızca bir snapshot tarihi var: {available_dates[0]}. "
            "Karşılaştırma için farklı günlerde tekrar snapshot alınması gerekir."
        )

        safe_plotly(chart_history_trend, history_summary_df)
        st.dataframe(history_summary_df, use_container_width=True, hide_index=True)
        return

    c1, c2, c3 = st.columns([1, 1, 1.1])

    with c1:
        start_date = st.selectbox("Başlangıç tarihi", available_dates, index=0)

    with c2:
        end_date = st.selectbox(
            "Bitiş tarihi",
            available_dates,
            index=len(available_dates) - 1,
        )

    if start_date > end_date:
        st.warning("Başlangıç tarihi bitiş tarihinden sonra olamaz.")
        return

    with c3:
        selected_history_metric = st.selectbox(
            "Trend metriği",
            [
                "record_count",
                "total_capacity",
                "total_occupancy",
                "avg_risk",
                "critical_count",
                "estimated_count",
            ],
            format_func=lambda x: {
                "record_count": "Kayıt Sayısı",
                "total_capacity": "Toplam Kapasite",
                "total_occupancy": "Mevcut Hayvan",
                "avg_risk": "Ortalama Risk",
                "critical_count": "Kritik Kayıt",
                "estimated_count": "Tahmini Veri İçeren Kayıt",
            }.get(x, x),
        )

    try:
        summary_compare = compare_summary(history_summary_df, start_date, end_date)
    except Exception:
        summary_compare = None

    if summary_compare:
        k1, k2, k3, k4, k5 = st.columns(5)

        k1.metric(
            "Kayıt Sayısı",
            int(summary_compare["record_count"]["new"]),
            int(summary_compare["record_count"]["delta"]),
        )

        k2.metric(
            "Toplam Kapasite",
            int(summary_compare["total_capacity"]["new"]),
            int(summary_compare["total_capacity"]["delta"]),
        )

        k3.metric(
            "Mevcut Hayvan",
            int(summary_compare["total_occupancy"]["new"]),
            int(summary_compare["total_occupancy"]["delta"]),
        )

        k4.metric(
            "Ortalama Risk",
            f"{summary_compare['avg_risk']['new']:.1f}",
            f"{summary_compare['avg_risk']['delta']:.1f}",
        )

        k5.metric(
            "Kritik Kayıt",
            int(summary_compare["critical_count"]["new"]),
            int(summary_compare["critical_count"]["delta"]),
        )

    safe_plotly(chart_history_trend, history_summary_df)

    try:
        fig = chart_history_metric(history_summary_df, selected_history_metric)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning("Seçilen tarihsel metrik grafiği oluşturulamadı.")
        with st.expander("Teknik hata detayı"):
            st.exception(e)

    try:
        compare_df = compare_snapshot_dates(history_df, start_date, end_date)
    except Exception:
        compare_df = pd.DataFrame()

    st.markdown("#### Merkez Bazlı Değişim")

    if compare_df.empty:
        st.warning("Seçilen tarihler için merkez bazlı karşılaştırma üretilemedi.")
    else:
        delta_metric = st.selectbox(
            "Merkez bazlı değişim grafiği",
            [
                "risk_score_delta",
                "occupancy_delta",
                "capacity_delta",
                "occupancy_rate_delta",
                "vet_count_delta",
            ],
            format_func=lambda x: {
                "risk_score_delta": "Risk Skoru Değişimi",
                "occupancy_delta": "Mevcut Hayvan Sayısı Değişimi",
                "capacity_delta": "Kapasite Değişimi",
                "occupancy_rate_delta": "Doluluk Oranı Değişimi",
                "vet_count_delta": "Veteriner Sayısı Değişimi",
            }.get(x, x),
        )

        try:
            st.plotly_chart(
                chart_record_delta(compare_df, delta_metric),
                use_container_width=True,
            )
        except Exception:
            st.warning("Merkez bazlı değişim grafiği oluşturulamadı.")

        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        st.download_button(
            label="Tarihsel Karşılaştırma CSV İndir",
            data=compare_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"smartshelter_compare_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_ai_analysis(filtered_df, history_summary_df, anomalies_df):
    section_header(
        "🤖 AI Analiz ve Karar Destek",
        "Harici AI API kullanmadan, kural tabanlı yönetici özeti, anomali tespiti ve senaryo simülasyonu üretir.",
    )

    try:
        ai_summary = generate_executive_summary(
            df=filtered_df,
            history_summary_df=history_summary_df,
            anomalies_df=anomalies_df,
        )
        st.markdown(ai_summary)
    except Exception as e:
        st.warning("Yönetici özeti üretilemedi.")
        with st.expander("Teknik hata detayı"):
            st.exception(e)

    st.divider()

    st.markdown("#### 🚨 AI Uyarılar / Anomali Tespiti")

    if anomalies_df is None or anomalies_df.empty:
        st.success("Belirgin anomali tespit edilmedi.")
    else:
        a1, a2, a3 = st.columns(3)

        a1.metric("Toplam Uyarı", len(anomalies_df))
        a2.metric(
            "Yüksek Uyarı",
            len(anomalies_df[anomalies_df["severity"] == "Yüksek"])
            if "severity" in anomalies_df.columns
            else 0,
        )
        a3.metric(
            "Kritik Uyarı",
            len(anomalies_df[anomalies_df["severity"] == "Kritik"])
            if "severity" in anomalies_df.columns
            else 0,
        )

        st.dataframe(anomalies_df, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("#### 🧪 Müdahale Senaryosu")

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        extra_capacity = st.number_input("+ Kapasite", min_value=0, max_value=5000, value=0, step=10)

    with s2:
        extra_vets = st.number_input("+ Veteriner", min_value=0, max_value=100, value=0, step=1)

    with s3:
        extra_adoptions = st.number_input("+ Sahiplendirme", min_value=0, max_value=5000, value=0, step=10)

    with s4:
        extra_sterilizations = st.number_input("+ Kısırlaştırma", min_value=0, max_value=5000, value=0, step=10)

    try:
        scenario_df = simulate_interventions(
            filtered_df,
            extra_capacity=extra_capacity,
            extra_vets=extra_vets,
            extra_adoptions=extra_adoptions,
            extra_sterilizations=extra_sterilizations,
        )
    except Exception as e:
        scenario_df = pd.DataFrame()
        st.warning("Senaryo simülasyonu çalıştırılamadı.")
        with st.expander("Teknik hata detayı"):
            st.exception(e)

    if scenario_df.empty:
        st.warning("Senaryo için risk analizine uygun kayıt bulunamadı.")
    else:
        avg_base_risk = scenario_df["base_risk_score"].mean()
        avg_scenario_risk = scenario_df["scenario_risk_score"].mean()
        avg_improvement = scenario_df["risk_score_improvement"].mean()

        c1, c2, c3 = st.columns(3)

        c1.metric("Mevcut Ortalama Risk", f"{avg_base_risk:.1f}")
        c2.metric("Senaryo Ortalama Risk", f"{avg_scenario_risk:.1f}")
        c3.metric("Ortalama İyileşme", f"{avg_improvement:.1f}")

        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

        st.download_button(
            label="Senaryo Sonucunu CSV İndir",
            data=scenario_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="smartshelter_scenario_simulation.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_source_management(
    candidate_resources_df,
    loaded_resources_info,
    excluded_df,
    df_all_loaded,
):
    section_header(
        "🧩 Kaynak Yönetimi ve Veri Ayrıştırma",
        "Bulunan, yüklenen ve ana analitikten ayrıştırılan kaynakların izlenmesi.",
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Aday Resource", len(candidate_resources_df))
    c2.metric("İçeri Alınan Resource", len(loaded_resources_info))
    c3.metric("Yüklenen Satır", len(df_all_loaded))
    c4.metric("Dışlanan Satır", len(excluded_df))

    if not candidate_resources_df.empty and "resource_category" in candidate_resources_df.columns:
        st.markdown("#### Resource Kategori Dağılımı")

        category_summary = (
            candidate_resources_df.groupby("resource_category", as_index=False)
            .agg(
                resource_count=("name", "count"),
                avg_relevance=("relevance_score", "mean"),
            )
            .sort_values("resource_count", ascending=False)
        )

        category_summary["avg_relevance"] = category_summary["avg_relevance"].round(1)

        st.dataframe(category_summary, use_container_width=True, hide_index=True)

    st.markdown("#### İçeri Alınan Resource Listesi")

    if loaded_resources_info.empty:
        st.info("İçeri alınan resource listesi yok.")
    else:
        display_cols = [
            "source_portal",
            "resource_category",
            "relevance_score",
            "package",
            "name",
            "format",
            "matched_query",
            "url",
        ]

        display_cols = [c for c in display_cols if c in loaded_resources_info.columns]

        st.dataframe(
            loaded_resources_info[display_cols],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Dışlanan Kayıtlar")

    if excluded_df.empty:
        st.success("Dışlanan kayıt yok.")
    else:
        display_cols = [
            "name",
            "city",
            "district",
            "source_portal",
            "source_resource",
            "resource_category",
            "data_scope",
            "analytics_exclusion_reason",
        ]

        display_cols = [c for c in display_cols if c in excluded_df.columns]

        st.warning(
            "Bu kayıtlar yüklenmiş ancak ana analitik/risk hesabı için uygun görülmemiştir."
        )

        st.dataframe(
            excluded_df[display_cols].head(500),
            use_container_width=True,
            hide_index=True,
        )


def render_report_downloads(
    df,
    district_summary,
    history_df,
    history_summary_df,
    anomalies_df,
    excluded_df,
):
    section_header(
        "📥 Rapor İndirme",
        "Filtrelenmiş veri, Excel raporu, anomaliler, dışlanan kayıtlar ve tarihsel snapshot dosyaları.",
    )

    if df.empty:
        st.warning("İndirilecek güncel kayıt bulunamadı.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.download_button(
                label="Güncel Filtrelenmiş CSV Rapor İndir",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="smartshelter_filtered_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with c2:
            st.download_button(
                label="Güncel Excel Rapor İndir",
                data=to_excel_bytes(df, district_summary),
                file_name="smartshelter_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    if anomalies_df is not None and not anomalies_df.empty:
        st.download_button(
            label="AI Anomali Raporu CSV İndir",
            data=anomalies_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="smartshelter_ai_anomalies.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if excluded_df is not None and not excluded_df.empty:
        st.download_button(
            label="Dışlanan Kayıtlar CSV İndir",
            data=excluded_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="smartshelter_excluded_records.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if history_df is not None and not history_df.empty:
        st.download_button(
            label="Tüm Tarihsel Snapshot CSV İndir",
            data=history_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="smartshelter_history_snapshots.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if history_summary_df is not None and not history_summary_df.empty:
        st.download_button(
            label="Tarihsel Özet CSV İndir",
            data=history_summary_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="smartshelter_history_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
inject_css()
render_hero()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("⚙️ Kontrol Paneli")

if st.sidebar.button("🔄 Cache Temizle", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

st.sidebar.header("Veri Kaynağı")

mode = st.sidebar.radio(
    "Veri modu",
    [
        "Stabil Demo CSV",
        "Canlı CKAN API Dene",
        "Türkiye Geneli CKAN Taraması",
    ],
)

raw_df = pd.DataFrame()
selected_source_name = "Stabil Demo CSV"
selected_resource_label = "Lokal CSV"
deep_scan = False
loaded_resources_info = pd.DataFrame()
candidate_resources_df = pd.DataFrame()
failed_resource_count = 0


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
if mode == "Stabil Demo CSV":
    try:
        raw_df = cached_load_local_data(LOCAL_FILE).copy()
        selected_source_name = "Stabil Demo CSV"
        selected_resource_label = "Lokal CSV"
        st.success("Stabil demo CSV verisi kullanılıyor.")
    except Exception as e:
        st.error("Lokal CSV dosyası okunamadı.")
        st.exception(e)
        st.stop()


elif mode == "Canlı CKAN API Dene":
    selected_source_name = st.sidebar.selectbox(
        "Canlı kaynak seç",
        list(CKAN_SOURCES.keys()),
    )

    source = CKAN_SOURCES[selected_source_name]

    deep_scan = st.sidebar.checkbox(
        "Derin geçmiş kaynak taraması",
        value=True,
    )

    rows = 100 if deep_scan else 20
    deep_queries_tuple = tuple(source.get("deep_queries", [])) if deep_scan else tuple()

    try:
        with st.spinner("CKAN resource listesi taranıyor..."):
            resources = cached_search_ckan_resources(
                source["base"],
                source["query"],
                rows,
                deep_queries_tuple,
            )

        candidate_resources_df = pd.DataFrame(resources)

        if not resources:
            raw_df, selected_source_name, selected_resource_label = fallback_local(
                "Bu kaynakta uygun resource bulunamadı. Lokal demo veri kullanılıyor."
            )

        else:
            def live_label(i):
                r = resources[i]
                return (
                    f"[{r.get('resource_category', '-')}] "
                    f"{r.get('package', '-')} | "
                    f"{r.get('name', '-')} | "
                    f"{str(r.get('format', '')).upper()} | "
                    f"Skor: {r.get('relevance_score', 0)}"
                )

            selected_resource_index = st.sidebar.selectbox(
                "Resource seç",
                options=list(range(len(resources))),
                format_func=live_label,
            )

            selected_resource = dict(resources[selected_resource_index])
            selected_resource["source_portal"] = selected_source_name

            selected_resource_label = live_label(selected_resource_index)

            with st.spinner("Seçilen resource yükleniyor..."):
                resource_tuple = to_resource_tuple(selected_resource)
                raw_df = cached_load_resource(resource_tuple).copy()

            if raw_df.empty:
                raw_df, selected_source_name, selected_resource_label = fallback_local(
                    "Seçilen canlı kaynak boş döndü. Lokal demo veri kullanılıyor."
                )
            else:
                raw_df["source_portal"] = selected_source_name
                raw_df["source_resource"] = selected_resource_label
                raw_df["source_url"] = selected_resource.get("url", "")
                raw_df["resource_category"] = selected_resource.get("resource_category", "")
                raw_df["relevance_score"] = selected_resource.get("relevance_score", 0)

                loaded_resources_info = pd.DataFrame([selected_resource])

                st.success(f"Canlı veri kaynağı yüklendi: {selected_source_name}")
                st.caption(selected_resource_label)

    except Exception as e:
        st.warning("Canlı veri çekilemedi. Lokal stabil veri kullanılıyor.")
        st.error(str(e))

        raw_df = cached_load_local_data(LOCAL_FILE).copy()
        selected_source_name = "Fallback Demo CSV"
        selected_resource_label = "Lokal CSV"


elif mode == "Türkiye Geneli CKAN Taraması":
    selected_source_name = "Türkiye Geneli CKAN Taraması"
    selected_resource_label = "Çoklu CKAN resource birleşimi"
    deep_scan = True

    st.sidebar.markdown("### 🇹🇷 Türkiye Geneli Tarama")

    rows_per_query = st.sidebar.slider(
        "Kaynak başına arama derinliği",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
    )

    max_resources_to_load = st.sidebar.slider(
        "İçeri alınacak maksimum resource",
        min_value=1,
        max_value=50,
        value=15,
        step=1,
    )

        auto_load = st.sidebar.checkbox(
        "Bulunan uygun kaynakları otomatik içeri al",
        value=True,
    )

    only_shelter_like = st.sidebar.checkbox(
        "Öncelik: barınak/bakımevi envanteri",
        value=True,
    )

    try:
        with st.spinner("Türkiye geneli CKAN portalları taranıyor..."):
            resources = cached_search_turkiye_ckan_resources(
                rows_per_query=rows_per_query,
            )

        candidate_resources_df = pd.DataFrame(resources)

        if not resources:
            raw_df, selected_source_name, selected_resource_label = fallback_local(
                "Türkiye geneli taramada uygun resource bulunamadı. Lokal demo veri kullanılıyor."
            )

        else:
            st.success(
                f"Türkiye geneli taramada {len(resources)} uygun resource adayı bulundu."
            )

            with st.expander("🇹🇷 Bulunan Resource Adayları", expanded=False):
                display_cols = [
                    "source_portal",
                    "resource_category",
                    "relevance_score",
                    "package",
                    "name",
                    "format",
                    "matched_query",
                    "package_modified",
                    "resource_last_modified",
                    "url",
                ]

                display_cols = [
                    c for c in display_cols
                    if c in candidate_resources_df.columns
                ]

                st.dataframe(
                    candidate_resources_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )

            if only_shelter_like:
                preferred_resources = [
                    r for r in resources
                    if r.get("resource_category") in [
                        "shelter_facility",
                        "general_animal",
                        "capacity",
                    ]
                ]

                if preferred_resources:
                    resources_for_selection = preferred_resources
                else:
                    resources_for_selection = resources
                    st.warning(
                        "Barınak/bakımevi öncelikli kaynak bulunamadı. "
                        "Tüm uygun adaylar kullanılacak."
                    )
            else:
                resources_for_selection = resources

            if auto_load:
                selected_resources = resources_for_selection[:max_resources_to_load]
            else:
                labels = [
                    (
                        f"[{r.get('resource_category', '-')}] "
                        f"{r.get('source_portal', '-')} | "
                        f"{r.get('package', '-')} | "
                        f"{r.get('name', '-')} | "
                        f"{str(r.get('format', '')).upper()} | "
                        f"Skor: {r.get('relevance_score', 0)}"
                    )
                    for r in resources_for_selection
                ]

                default_count = min(max_resources_to_load, len(labels))

                selected_indices = st.sidebar.multiselect(
                    "İçeri alınacak resource seç",
                    options=list(range(len(resources_for_selection))),
                    default=list(range(default_count)),
                    format_func=lambda i: labels[i],
                )

                selected_resources = [
                    resources_for_selection[i]
                    for i in selected_indices
                ]

            if not selected_resources:
                raw_df, selected_source_name, selected_resource_label = fallback_local(
                    "İçeri alınacak resource seçilmedi. Lokal demo veri kullanılıyor."
                )
            else:
                selected_resources_for_cache = tuple(
                    to_resource_tuple(r) for r in selected_resources
                )

                with st.spinner("Seçilen resource dosyaları içeri alınıyor..."):
                    raw_df, loaded_resources, failed_resources = cached_load_multiple_resources(
                        selected_resources_for_cache,
                        max_resources=max_resources_to_load,
                    )

                raw_df = raw_df.copy()
                failed_resource_count = len(failed_resources)
                loaded_resources_info = pd.DataFrame(loaded_resources)

                if raw_df.empty:
                    raw_df, selected_source_name, selected_resource_label = fallback_local(
                        "Seçilen resource dosyaları okunamadı veya boş geldi. "
                        "Lokal demo veri kullanılıyor."
                    )
                else:
                    st.success(
                        f"{len(loaded_resources)} resource başarıyla içeri alındı. "
                        f"{failed_resource_count} resource okunamadı/boş geldi."
                    )

                    with st.expander("İçeri Alınan Resource Listesi", expanded=False):
                        display_cols = [
                            "source_portal",
                            "resource_category",
                            "relevance_score",
                            "package",
                            "name",
                            "format",
                            "matched_query",
                            "url",
                        ]

                        display_cols = [
                            c for c in display_cols
                            if c in loaded_resources_info.columns
                        ]

                        st.dataframe(
                            loaded_resources_info[display_cols],
                            use_container_width=True,
                            hide_index=True,
                        )

    except Exception as e:
        st.warning("Türkiye geneli tarama başarısız oldu. Lokal demo veri kullanılıyor.")
        st.error(str(e))

        raw_df = cached_load_local_data(LOCAL_FILE).copy()
        selected_source_name = "Fallback Demo CSV"
        selected_resource_label = "Lokal CSV"


if raw_df.empty:
    st.error("Veri yüklenemedi.")
    st.stop()


# ---------------------------------------------------------
# Normalize + Eligibility
# ---------------------------------------------------------
try:
    df_all_loaded = normalize_columns(raw_df)
    df_all_loaded = ensure_app_columns(df_all_loaded)
except Exception as e:
    st.error("Veri normalize edilirken hata oluştu.")
    st.exception(e)
    st.stop()

excluded_df = pd.DataFrame()

if "analytics_eligible" in df_all_loaded.columns:
    eligible_mask = as_bool_series(df_all_loaded["analytics_eligible"])

    excluded_df = df_all_loaded[~eligible_mask].copy()
    df = df_all_loaded[eligible_mask].copy()
else:
    df = df_all_loaded.copy()

if df.empty:
    st.warning(
        "Yüklenen kaynaklar ana envanter/analitik ekranı için uygun görünmüyor. "
        "Lokal demo veriye dönülüyor."
    )

    raw_df = cached_load_local_data(LOCAL_FILE).copy()
    selected_source_name = "Fallback Demo CSV"
    selected_resource_label = "Lokal CSV - analitik uygun kaynak bulunamadı"

    df_all_loaded = normalize_columns(raw_df)
    df_all_loaded = ensure_app_columns(df_all_loaded)
    excluded_df = pd.DataFrame()
    df = df_all_loaded.copy()


# ---------------------------------------------------------
# Risk + AI + Data Quality
# ---------------------------------------------------------
try:
    df = calculate_risk(df)
    df = ensure_app_columns(df)

    if "risk_eligible" in df.columns:
        not_risk_ready = ~as_bool_series(df["risk_eligible"])
        df.loc[not_risk_ready, "risk_level"] = "Veri yetersiz"

    df = create_action_recommendations(df)
    df = ensure_app_columns(df)

    df = calculate_data_quality_score(df)
    df = ensure_app_columns(df)

    df = generate_risk_explanations(df)
    df = ensure_app_columns(df)

except Exception as e:
    st.error("Risk / veri kalitesi / AI açıklama adımlarında hata oluştu.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# Snapshot + History
# ---------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("Snapshot")

snapshot_mode = st.sidebar.radio(
    "Tarihsel kayıt",
    ["Mevcut geçmişi kullan", "Bugünün snapshot'unu kaydet/güncelle"],
    index=0,
)

history_df = load_history_file()

if snapshot_mode == "Bugünün snapshot'unu kaydet/güncelle" or history_df.empty:
    try:
        history_df = append_snapshot(
            df=df,
            source_name=selected_source_name,
            resource_label=selected_resource_label,
        )
        st.sidebar.success(f"Snapshot hazır: {date.today().isoformat()}")
    except Exception as e:
        st.sidebar.warning("Snapshot yazılamadı. Mevcut geçmiş kullanılacak.")
        with st.sidebar.expander("Snapshot hata detayı"):
            st.exception(e)
        history_df = load_history_file()

try:
    history_summary_df = (
        build_history_summary(history_df)
        if history_df is not None and not history_df.empty
        else pd.DataFrame()
    )
except Exception:
    history_summary_df = pd.DataFrame()

try:
    anomalies_df = detect_anomalies(
        history_df=history_df,
        current_df=df,
    )
except Exception:
    anomalies_df = pd.DataFrame()


# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("Filtreler")

cities = sorted(df["city"].dropna().astype(str).unique().tolist())
districts = sorted(df["district"].dropna().astype(str).unique().tolist())

risk_order = ["Düşük", "Orta", "Kritik", "Veri yetersiz"]
available_risks = df["risk_level"].dropna().astype(str).unique().tolist()
risk_levels = [r for r in risk_order if r in available_risks]
risk_levels += [r for r in available_risks if r not in risk_levels]

use_city_filter = st.sidebar.checkbox("İl filtresi kullan", value=False)

if use_city_filter:
    selected_cities = st.sidebar.multiselect(
        "İl seç",
        cities,
        default=cities,
    )
else:
    selected_cities = cities

use_district_filter = st.sidebar.checkbox("İlçe filtresi kullan", value=False)

if use_district_filter:
    selected_districts = st.sidebar.multiselect(
        "İlçe seç",
        districts,
        default=districts,
    )
else:
    selected_districts = districts

selected_risks = st.sidebar.multiselect(
    "Risk seviyesi",
    risk_levels,
    default=risk_levels,
)

show_only_valid_coordinates = st.sidebar.checkbox(
    "Sadece koordinatı geçerli kayıtlar",
    value=False,
)

show_only_risk_ready = st.sidebar.checkbox(
    "Sadece risk analizine uygun kayıtlar",
    value=False,
)

data_scope_values = sorted(df["data_scope"].dropna().astype(str).unique().tolist())

selected_scopes = st.sidebar.multiselect(
    "Veri kapsamı",
    data_scope_values,
    default=data_scope_values,
)

filtered_df = df[
    (df["city"].astype(str).isin(selected_cities))
    & (df["district"].astype(str).isin(selected_districts))
    & (df["risk_level"].astype(str).isin(selected_risks))
    & (df["data_scope"].astype(str).isin(selected_scopes))
].copy()

if show_only_valid_coordinates:
    filtered_df = filtered_df[
        as_bool_series(filtered_df["coordinate_valid"])
    ].copy()

if show_only_risk_ready:
    filtered_df = filtered_df[
        as_bool_series(filtered_df["risk_eligible"])
    ].copy()

try:
    district_summary = build_district_summary(filtered_df)
except Exception:
    district_summary = pd.DataFrame()


# ---------------------------------------------------------
# Top Info + KPIs
# ---------------------------------------------------------
render_data_source_status(
    selected_source_name=selected_source_name,
    selected_resource_label=selected_resource_label,
    deep_scan=deep_scan,
    candidate_resources_df=candidate_resources_df,
    loaded_resources_info=loaded_resources_info,
    df_all_loaded=df_all_loaded,
    excluded_df=excluded_df,
    failed_resource_count=failed_resource_count,
    mode=mode,
)

section_header(
    "📌 Anlık Durum",
    "Seçili filtrelere göre güncel kapasite, risk ve veri kapsamı özeti.",
)

render_kpis(filtered_df)

st.divider()


# ---------------------------------------------------------
# Map + Detail
# ---------------------------------------------------------
left, right = st.columns([2.25, 1], gap="large")

with left:
    section_header(
        "📍 GIS Haritası",
        "Koordinatı geçerli kayıtlar risk durumuna göre harita üzerinde gösterilir.",
    )

    if "coordinate_valid" in filtered_df.columns:
        map_df = filtered_df[
            as_bool_series(filtered_df["coordinate_valid"])
        ].copy()
    else:
        map_df = pd.DataFrame()

    if map_df.empty:
        st.warning(
            "Harita için geçerli koordinata sahip kayıt bulunamadı. "
            "Türkiye geneli kaynakların çoğunda koordinat alanı olmayabilir."
        )
    else:
        try:
            shelter_map = create_shelter_map(map_df)

            st_folium(
                shelter_map,
                width=1100,
                height=620,
                returned_objects=[],
            )
        except Exception as e:
            st.warning("Harita oluşturulamadı.")
            with st.expander("Teknik hata detayı"):
                st.exception(e)

with right:
    render_record_detail(filtered_df)

st.divider()


# ---------------------------------------------------------
# Dashboard Tabs
# ---------------------------------------------------------
section_header(
    "📊 Dashboard",
    "Risk, doluluk, ilçe özeti, AI analiz, veri kalitesi ve rapor modülleri.",
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
    [
        "⚠️ Risk",
        "🏠 Doluluk",
        "🗺️ İlçe Özeti",
        "🕒 Geçmiş",
        "🤖 AI Analiz",
        "🧩 Kaynaklar",
        "🧪 Veri Kalitesi",
        "🧾 Ham Veri",
        "📥 Rapor",
        "🌍 Vizyon",
    ]
)


# ---------------------------------------------------------
# Tab 1 - Risk
# ---------------------------------------------------------
with tab1:
    section_header(
        "⚠️ Risk Skoru",
        "Risk analizine uygun kayıtlar için önceliklendirme ve operasyonel aksiyon listesi.",
    )

    if filtered_df.empty:
        st.warning("Filtreye uygun kayıt bulunamadı.")
    else:
        safe_plotly(chart_risk_score, filtered_df)

        st.markdown("#### Operasyonel Öncelik Listesi")

        priority_cols = [
            "name",
            "city",
            "district",
            "source_portal",
            "resource_category",
            "data_scope",
            "risk_eligible",
            "risk_level",
            "risk_score",
            "occupancy_rate",
            "animals_per_vet",
            "data_quality_score",
            "recommended_action",
            "risk_explanation",
        ]

        priority_cols = [c for c in priority_cols if c in filtered_df.columns]

        sort_col = (
            "risk_score" if "risk_score" in filtered_df.columns else priority_cols[0]
        )

        st.dataframe(
            filtered_df.sort_values(
                sort_col,
                ascending=False,
                na_position="last",
            )[priority_cols],
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# Tab 2 - Occupancy
# ---------------------------------------------------------
with tab2:
    section_header(
        "🏠 Doluluk ve Kapasite",
        "Bilinen kapasite ve mevcut hayvan sayısı üzerinden doluluk baskısının izlenmesi.",
    )

    if filtered_df.empty:
        st.warning("Filtreye uygun kayıt bulunamadı.")
    else:
        safe_plotly(chart_occupancy_rate, filtered_df)

        st.markdown("#### Doluluk Detay Tablosu")

        occupancy_cols = [
            "name",
            "city",
            "district",
            "capacity_available",
            "occupancy_available",
            "capacity",
            "occupancy",
            "occupancy_rate",
            "risk_level",
            "risk_score",
            "source_portal",
        ]

        occupancy_cols = [c for c in occupancy_cols if c in filtered_df.columns]

        st.dataframe(
            filtered_df.sort_values(
                "occupancy_rate",
                ascending=False,
                na_position="last",
            )[occupancy_cols],
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# Tab 3 - District Summary
# ---------------------------------------------------------
with tab3:
    section_header(
        "🗺️ İlçe Bazlı Özet",
        "İlçe düzeyinde ortalama risk, kapasite ve kayıt dağılımı.",
    )

    if district_summary.empty:
        st.warning("İlçe özeti oluşturulamadı.")
    else:
        safe_plotly(chart_district_avg_risk, district_summary)

        st.markdown("#### İlçe Bazlı Özet Tablosu")

        st.dataframe(
            district_summary,
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# Tab 4 - Historical Analytics
# ---------------------------------------------------------
with tab4:
    render_history_analytics(
        history_df,
        history_summary_df,
    )


# ---------------------------------------------------------
# Tab 5 - AI Analysis
# ---------------------------------------------------------
with tab5:
    render_ai_analysis(
        filtered_df=filtered_df,
        history_summary_df=history_summary_df,
        anomalies_df=anomalies_df,
    )


# ---------------------------------------------------------
# Tab 6 - Source Management
# ---------------------------------------------------------
with tab6:
    render_source_management(
        candidate_resources_df=candidate_resources_df,
        loaded_resources_info=loaded_resources_info,
        excluded_df=excluded_df,
        df_all_loaded=df_all_loaded,
    )


# ---------------------------------------------------------
# Tab 7 - Data Quality
# ---------------------------------------------------------
with tab7:
    render_data_quality_summary(filtered_df)

    st.markdown("#### Veri Kalitesi Detayı")

    quality_cols = [
        "name",
        "city",
        "district",
        "source_portal",
        "source_resource",
        "resource_category",
        "data_scope",
        "risk_eligible",
        "data_quality_score",
        "data_quality_level",
        "coordinate_valid",
        "analytics_eligible",
        "capacity_available",
        "occupancy_available",
        "vet_count_available",
        "data_quality_note",
    ]

    quality_cols = [c for c in quality_cols if c in filtered_df.columns]

    if filtered_df.empty:
        st.warning("Filtreye uygun kayıt bulunamadı.")
    else:
        st.dataframe(
            filtered_df.sort_values(
                "data_quality_score",
                ascending=True,
                na_position="last",
            )[quality_cols],
            use_container_width=True,
            hide_index=True,
        )

    if not excluded_df.empty:
        st.markdown("#### Dışlanan Kayıtlar")

        excluded_cols = [
            "name",
            "city",
            "district",
            "source_portal",
            "source_resource",
            "resource_category",
            "data_scope",
            "analytics_exclusion_reason",
            "data_quality_note",
        ]

        excluded_cols = [c for c in excluded_cols if c in excluded_df.columns]

        st.warning(
            "Aşağıdaki kayıtlar yüklendi ancak ana analitik için uygun görülmedi."
        )

        st.dataframe(
            excluded_df[excluded_cols].head(500),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# Tab 8 - Raw Data
# ---------------------------------------------------------
with tab8:
    section_header(
        "🧾 Ham ve Normalize Veri",
        "Filtrelenmiş analitik veri seti ve tüm normalize edilmiş kayıtlar.",
    )

    st.markdown("#### Filtrelenmiş Analitik Veri Seti")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Tüm Yüklenen Normalize Veri")

    st.caption(
        "Bu tablo, ana analitiğe dahil edilen ve dışlanan tüm normalize edilmiş kayıtları gösterir."
    )

    st.dataframe(
        df_all_loaded,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Tab 9 - Reports
# ---------------------------------------------------------
with tab9:
    render_report_downloads(
        df=filtered_df,
        district_summary=district_summary,
        history_df=history_df,
        history_summary_df=history_summary_df,
        anomalies_df=anomalies_df,
        excluded_df=excluded_df,
    )


# ---------------------------------------------------------
# Tab 10 - Vision
# ---------------------------------------------------------
with tab10:
    section_header(
        "🌍 SmartShelter GIS Vizyonu",
        "Açık veri, GIS, risk önceliklendirme ve karar destek yaklaşımı.",
    )

    st.markdown(
        """
        SmartShelter GIS; belediyeler, Tarım ve Orman Bakanlığı, STK'lar ve yerel yönetimler arasında
        ortak bir veri dili oluşturmayı hedefleyen açık veri tabanlı bir GIS karar destek prototipidir.

        #### Temel Amaçlar

        - Hayvan bakımevleri ve geçici bakım merkezlerini harita üzerinde izlemek
        - Kapasite, doluluk ve operasyonel baskıyı görünür hale getirmek
        - Veteriner iş yükünü ölçülebilir hale getirmek
        - Kısırlaştırma ve sahiplendirme performansını takip etmek
        - Kritik merkezleri önceliklendirmek
        - İl/ilçe bazlı operasyonel planlamaya destek olmak
        - Geçmiş veriler üzerinden değişim analizi yapmak
        - Türkiye geneli açık veri kaynaklarını sınıflandırarak analiz etmek

        #### Veri Ayrıştırma Yaklaşımı

        Türkiye geneli açık veri portallarında **hayvan**, **veteriner**, **barınak**,
        **bakımevi** gibi ifadeler içeren çok sayıda farklı veri seti bulunabilir.
        Bu veri setlerinin tamamı doğrudan risk analizi için uygun değildir.

        Bu nedenle sistem kayıtları şu veri kapsamlarına ayırır:

        - **risk_ready:** Kapasite, mevcut hayvan ve operasyonel alanları yeterli olan kayıtlar
        - **capacity_only:** Kapasite veya tesis bilgisi var ama risk için eksik veri bulunan kayıtlar
        - **location_only:** Konum/envanter bilgisi olan ancak operasyonel metrikleri eksik kayıtlar
        - **operation_stats:** İşlem sayısı, yıllık istatistik, denetim gibi operasyonel tablolar
        - **unknown:** Sınıflandırılamayan veya yetersiz kayıtlar

        Risk dashboard'u yalnızca uygun kayıtları risk hesabında kullanır.
        Eksik veya kapsam dışı kaynaklar veri yönetimi ve kalite sekmelerinde ayrıca gösterilir.

        #### AI Destekli Karar Destek Yaklaşımı

        Bu prototipteki AI modülü harici model kullanmadan, şeffaf ve kural tabanlı olarak çalışır.
        Sistem şu çıktıları üretir:

        - Yönetici özeti
        - Risk açıklaması
        - Veri kalite skoru
        - Anomali tespiti
        - Müdahale senaryosu simülasyonu
        - Operasyonel öncelik listesi

        #### Geçmiş Analitik Yaklaşımı

        Sistem isteğe bağlı olarak günlük snapshot oluşturur. Böylece şu sorular yanıtlanabilir:

        - Toplam kapasite önceki tarihe göre arttı mı?
        - Mevcut hayvan sayısı azaldı mı, arttı mı?
        - Ortalama risk hangi tarihte yükseldi?
        - Hangi merkezde risk skoru kötüleşti?
        - Hangi merkez yeni eklendi veya veri kaynağından çıktı?
        - Veri kalitesi zamanla iyileşti mi?

        #### Önerilen Sonraki Aşamalar

        1. Ulusal veri standardı oluşturulması  
        2. Belediye sistemleriyle API entegrasyonu  
        3. PostGIS tabanlı merkezi coğrafi veri altyapısı  
        4. Mobil saha veri girişi  
        5. Gerçek zamanlı kapasite ve vaka takibi  
        6. Bakanlık düzeyinde izleme ve raporlama ekranı  
        7. Zaman serisi tabanlı erken uyarı sistemi  
        8. CKAN dışı belediye web sayfaları için kontrollü scraping modülü  
        9. LLM destekli doğal dil sorgulama ve otomatik raporlama  

        > Not: Bu uygulama resmi bir sistem değil, karar destek amaçlı çalışan prototip bir yazılımdır.
        """
    )


st.info(
    "Canlı API erişimi başarısız olursa sistem otomatik olarak lokal stabil CSV verisine döner. "
    "Türkiye geneli taramada farklı kapsamlı kaynaklar ayrıştırılır; risk hesabına yalnızca uygun kayıtlar dahil edilir. "
    "Snapshot kaydı sidebar üzerinden kontrollü şekilde yapılır."
)
