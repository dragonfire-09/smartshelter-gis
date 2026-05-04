"""
SmartShelter GIS - Ana Uygulama
================================
Hayvan bakımevi izleme, risk önceliklendirme ve GIS karar destek paneli.
"""
from io import BytesIO
from pathlib import Path

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
    search_turkiye_ckan_resources_with_progress,
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

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="SmartShelter GIS",
    page_icon="🐾",
    layout="wide",
)

HISTORY_FILE = Path("data/history/shelter_history.csv")
RISK_LEVEL_ORDER = ["Düşük", "Orta", "Yüksek", "Kritik", "Veri yetersiz"]


# =========================================================
# CACHED LOADERS
# =========================================================
@st.cache_data(show_spinner=False)
def cached_load_local_data(path):
    return load_local_data(path)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_search_turkiye_ckan_resources(rows_per_query):
    """Cache'li, progress bar OLMADAN tarama (fallback için)."""
    return search_turkiye_ckan_resources(rows_per_query=rows_per_query)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_load_multiple_resources(resources_tuple, max_resources):
    resources = [dict(items) for items in resources_tuple]
    return load_multiple_resources(resources, max_resources=max_resources)


def to_resource_tuple(resource: dict) -> tuple:
    safe_items = []
    for k, v in resource.items():
        if isinstance(v, (list, dict, set)):
            safe_items.append((k, str(v)))
        else:
            safe_items.append((k, v))
    return tuple(sorted(safe_items, key=lambda kv: kv[0]))


# =========================================================
# UI HELPERS
# =========================================================
def inject_css():
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 1.2rem; max-width: 1500px; }
        .hero {
            background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
            border-radius: 12px; padding: 1.5rem; color: white; margin-bottom: 1rem;
        }
        .hero-title { font-size: 1.8rem; font-weight: 700; }
        .hero-subtitle { opacity: 0.85; margin-top: 0.5rem; }
        .hero-badges { margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .pill {
            background: rgba(255,255,255,0.12); padding: 0.3rem 0.8rem;
            border-radius: 999px; font-size: 0.85rem;
        }
        .section-title { font-size: 1.3rem; font-weight: 600; margin-top: 1rem; }
        .section-caption { color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.8rem; }
        .progress-card {
            background: #f1f5f9; border-left: 4px solid #3b82f6;
            padding: 0.8rem 1rem; border-radius: 8px; margin: 0.5rem 0;
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
                <span class="pill">🧠 AI Analiz</span>
                <span class="pill">📊 CKAN Açık Veri</span>
                <span class="pill">🔒 Strict Mode</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def as_bool_series(series):
    if series is None:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes", "evet", "var", "available"])
    )


def ensure_app_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Eksik sütunları güvenli default'larla doldur ve flag'leri compute et."""
    df = df.copy()

    string_defaults = {
        "name": pd.NA, "city": pd.NA, "district": pd.NA,
        "source_portal": "", "source_resource": "", "source_url": "",
        "resource_category": "unknown", "data_scope": "unknown",
        "risk_level": "Veri yetersiz", "recommended_action": "",
        "risk_explanation": "", "data_quality_level": "Bilinmiyor",
        "data_quality_note": "", "analytics_exclusion_reason": "",
    }
    for col, default in string_defaults.items():
        if col not in df.columns:
            df[col] = default

    numeric_cols = [
        "capacity", "occupancy", "vet_count", "sterilization_count",
        "adoption_count", "occupancy_rate", "animals_per_vet",
        "risk_score", "data_quality_score", "lat", "lon", "latitude", "longitude",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # lat/lon ↔ latitude/longitude eşitle
    if df["lat"].isna().all() and not df["latitude"].isna().all():
        df["lat"] = df["latitude"]
    if df["lon"].isna().all() and not df["longitude"].isna().all():
        df["lon"] = df["longitude"]

    bool_flags = [
        "name_available", "city_available", "capacity_available",
        "occupancy_available", "vet_count_available", "coordinate_valid",
        "analytics_eligible", "risk_eligible",
    ]
    for flag in bool_flags:
        if flag not in df.columns:
            df[flag] = False
        else:
            df[flag] = as_bool_series(df[flag])

    # Otomatik flag'leri compute et
    df["name_available"] = (
        df["name"].astype(str).str.strip()
        .replace({"nan": "", "None": "", "<NA>": ""})
        .ne("")
    )
    df["coordinate_valid"] = (
        df["lat"].between(-90, 90, inclusive="both")
        & df["lon"].between(-180, 180, inclusive="both")
    )
    df["capacity_available"] = df["capacity"].notna() & (df["capacity"] > 0)
    df["occupancy_available"] = df["occupancy"].notna()
    df["risk_eligible"] = df["capacity_available"] & df["occupancy_available"]

    # data_scope sınıflandır
    def classify_scope(row):
        if row["risk_eligible"] and row["coordinate_valid"]:
            return "risk_ready"
        if row["risk_eligible"]:
            return "capacity_only"
        if row["coordinate_valid"]:
            return "location_only"
        return "metadata_only"
    df["data_scope"] = df.apply(classify_scope, axis=1)

    return df


# =========================================================
# CKAN TARAMA — PROGRESS BAR İLE
# =========================================================
def search_turkiye_with_progress(rows_per_query):
    """Türkiye geneli CKAN taramasını canlı progress bar ile yapar."""
    # Session state ile cache
    cache_key = f"ckan_search_cache_{rows_per_query}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # UI elementleri
    progress_container = st.container()
    with progress_container:
        st.markdown("#### 🇹🇷 Türkiye Geneli CKAN Taraması")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        portal_count_text = st.empty()

    total_portals = len(TURKIYE_CKAN_SOURCES) if TURKIYE_CKAN_SOURCES else 1
    found_resources = []
    portal_progress = {"done": 0, "found": 0}

    def progress_callback(portal_name=None, current=0, total=total_portals, found=0, message=""):
        """data_loader'dan çağrılan callback."""
        try:
            ratio = min(max(current / max(total, 1), 0.0), 1.0)
            progress_bar.progress(ratio)
            
            label = portal_name or "Portal"
            if message:
                status_text.info(f"🔍 **{label}** · {message}")
            else:
                status_text.info(f"🔍 Taranan: **{label}** ({current}/{total})")
            
            portal_count_text.caption(
                f"📊 İlerleme: {current}/{total} portal · {found} resource bulundu"
            )
        except Exception:
            pass

    try:
        # Önce progress bar'lı versiyonu dene
        result = search_turkiye_ckan_resources_with_progress(
            rows_per_query=rows_per_query,
            progress_callback=progress_callback,
        )
    except TypeError:
        # Eski imza — callback parametresi yok
        try:
            result = search_turkiye_ckan_resources_with_progress(rows_per_query=rows_per_query)
        except Exception:
            result = None
    except Exception as e:
        status_text.error(f"❌ Tarama hatası: {e}")
        result = None

    # Fallback: progress'siz versiyon
    if result is None:
        status_text.warning("⏳ Progress destekli tarama başarısız, klasik moda geçiliyor...")
        try:
            result = cached_search_turkiye_ckan_resources(rows_per_query)
        except Exception as e:
            status_text.error(f"❌ CKAN tarama hatası: {e}")
            result = []

    # Bitiş animasyonu
    progress_bar.progress(1.0)

    # Result'u DataFrame'e çevir
    if isinstance(result, pd.DataFrame):
        df = result.copy()
    elif isinstance(result, list):
        df = pd.DataFrame(result) if result else pd.DataFrame()
    else:
        df = pd.DataFrame()

    status_text.success(f"✅ Tarama tamamlandı: **{len(df)}** uygun resource bulundu.")
    portal_count_text.empty()

    # Cache'le
    st.session_state[cache_key] = df
    return df


# =========================================================
# SIDEBAR
# =========================================================
def render_sidebar():
    st.sidebar.title("⚙️ Kontrol Paneli")

    if st.sidebar.button("🔄 Cache Temizle", width="stretch"):
        st.cache_data.clear()
        # Session state'deki tarama cache'lerini de temizle
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith("ckan_search_cache_")]
        for k in keys_to_remove:
            del st.session_state[k]
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Veri Kaynağı")
    mode = st.sidebar.radio(
        "Veri modu",
        ["Stabil Demo CSV", "Türkiye Geneli CKAN Taraması"],
        index=0,
    )

    rows_per_query = 50
    max_resources = 15
    if mode == "Türkiye Geneli CKAN Taraması":
        st.sidebar.markdown("### 🇹🇷 Türkiye Geneli Tarama")
        rows_per_query = st.sidebar.slider("Kaynak başına arama derinliği", 10, 100, 50, 10)
        max_resources = st.sidebar.slider("Maksimum resource", 5, 30, 15, 1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔒 Veri Kalitesi Modu")
    strict_mode = st.sidebar.toggle(
        "Strict Mode (Sadece Gerçek Veri)",
        value=True,
        help="Sadece kapasite/mevcut hayvan alanı dolu kayıtları kullanır.",
    )

    return mode, rows_per_query, max_resources, strict_mode


# =========================================================
# DATA PIPELINE (Hibrit Mod + Progress)
# =========================================================
def load_pipeline(mode, rows_per_query, max_resources):
    """Veri yükleme: Demo CSV her zaman temel, CKAN üstüne ekleniyor."""
    candidate_df = pd.DataFrame()
    loaded_info = []

    # Demo CSV her zaman temel
    demo_raw = cached_load_local_data(LOCAL_FILE).copy()
    demo_df = normalize_columns(demo_raw)
    demo_df = ensure_app_columns(demo_df)
    demo_df["source_portal"] = "Demo CSV"

    if mode == "Stabil Demo CSV":
        return demo_df, candidate_df, loaded_info, "Stabil Demo CSV", "data/kocaeli_shelters.csv"

    # ---- CKAN modu: progress bar ile tarama ----
    candidate_df = search_turkiye_with_progress(rows_per_query)

    if candidate_df.empty:
        st.warning("CKAN taramasında uygun resource bulunamadı. Demo veri kullanılıyor.")
        return demo_df, candidate_df, loaded_info, "Stabil Demo CSV (CKAN boş)", "data/kocaeli_shelters.csv"

    candidate_records = candidate_df.to_dict(orient="records")

    st.success(f"Türkiye geneli taramada {len(candidate_df)} uygun resource adayı bulundu.")
    with st.expander("🇹🇷 Bulunan Resource Adayları", expanded=False):
        st.dataframe(candidate_df, width="stretch", height=300)

    # ---- Resource yükleme: progress bar ile ----
    st.markdown("#### 📥 Resource'lar İndiriliyor")
    download_progress = st.progress(0.0)
    download_status = st.empty()
    download_status.info(f"⏳ {min(max_resources, len(candidate_records))} resource yüklenecek...")

    resources_tuple = tuple(to_resource_tuple(r) for r in candidate_records)

    try:
        result = cached_load_multiple_resources(resources_tuple, max_resources)
        if isinstance(result, tuple) and len(result) >= 2:
            df_loaded = result[0]
            loaded_info = result[1]
        elif isinstance(result, pd.DataFrame):
            df_loaded = result
            loaded_info = []
        else:
            df_loaded = pd.DataFrame()
            loaded_info = []
    except Exception as e:
        download_status.error(f"❌ Resource yükleme hatası: {e}")
        df_loaded = pd.DataFrame()
        loaded_info = []

    download_progress.progress(1.0)
    download_status.success(
        f"✅ {len(loaded_info) if loaded_info else 0} resource başarıyla yüklendi · "
        f"{len(df_loaded)} satır."
    )

    if df_loaded.empty:
        st.warning("CKAN'dan veri yüklenemedi. Demo veri kullanılıyor.")
        return demo_df, candidate_df, loaded_info, "Stabil Demo CSV (CKAN okunamadı)", "data/kocaeli_shelters.csv"

    df_loaded = normalize_columns(df_loaded)
    df_loaded = ensure_app_columns(df_loaded)
    if "source_portal" not in df_loaded.columns or df_loaded["source_portal"].astype(str).eq("").all():
        df_loaded["source_portal"] = "CKAN"

    # Hibrit mod: Demo + CKAN her zaman birleştirilir
    usable_ckan = int(df_loaded["coordinate_valid"].sum())
    has_cap_ckan = int(df_loaded["capacity_available"].sum())

    st.info(
        f"📊 CKAN'dan {len(df_loaded)} satır yüklendi "
        f"({usable_ckan} koordinatlı, {has_cap_ckan} kapasite bilgili). "
        f"Demo verisi ({len(demo_df)} kayıt) ile birleştiriliyor."
    )

    combined = pd.concat([demo_df, df_loaded], ignore_index=True)
    combined = ensure_app_columns(combined)

    return (
        combined,
        candidate_df,
        loaded_info,
        "Hibrit (Demo + CKAN)",
        f"Demo + {len(loaded_info) if loaded_info else len(df_loaded)} CKAN resource",
    )


# =========================================================
# RENDER BLOCKS
# =========================================================
def render_kpis(df: pd.DataFrame):
    section_header("📌 Anlık Durum", "Seçili filtrelere göre güncel kapasite, risk ve veri kapsamı özeti.")

    if df.empty:
        st.warning("Filtreye uygun kayıt bulunamadı.")
        return

    cap_av = as_bool_series(df["capacity_available"])
    occ_av = as_bool_series(df["occupancy_available"])
    risk_ok = as_bool_series(df["risk_eligible"])
    coord_ok = as_bool_series(df["coordinate_valid"])

    known_cap = int(df.loc[cap_av, "capacity"].fillna(0).sum())
    known_occ = int(df.loc[occ_av, "occupancy"].fillna(0).sum())
    risk_df = df[risk_ok]
    avg_risk = risk_df["risk_score"].mean() if len(risk_df) else None
    critical = (df["risk_level"].astype(str) == "Kritik").sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Envanter Kaydı", len(df))
    c2.metric("Risk Hazır", int(risk_ok.sum()))
    c3.metric("Bilinen Kapasite", known_cap)
    c4.metric("Bilinen Mevcut", known_occ)
    c5.metric("Ortalama Risk", f"{avg_risk:.1f}" if avg_risk is not None else "—")
    c6.metric("Geçerli Koordinat", int(coord_ok.sum()))

    if critical > 0:
        st.error(f"🚨 {critical} kritik kayıt mevcut.")
    else:
        st.success("🟢 Kritik seviyede kayıt yok.")


def render_map(df: pd.DataFrame):
    section_header(
        "📍 GIS Haritası",
        "Koordinatı geçerli ve isim alanı dolu kayıtlar risk durumuna göre haritada gösterilir.",
    )

    map_df = df[
        as_bool_series(df["coordinate_valid"])
        & as_bool_series(df["name_available"])
    ].copy()

    if map_df.empty:
        st.warning("Harita için geçerli koordinata sahip ve isim alanı dolu kayıt bulunamadı.")
        with st.expander("🔍 Teşhis"):
            st.write(f"Toplam kayıt: {len(df)}")
            st.write(f"Geçerli koordinatlı: {int(as_bool_series(df['coordinate_valid']).sum())}")
            st.write(f"İsim alanı dolu: {int(as_bool_series(df['name_available']).sum())}")
            if not df.empty:
                cols = [c for c in ["name", "city", "district", "lat", "lon"] if c in df.columns]
                if cols:
                    st.dataframe(df[cols].head(10), width="stretch")
        return

    try:
        shelter_map = create_shelter_map(map_df)
        st_folium(shelter_map, width=1100, height=620, returned_objects=[])
    except Exception as e:
        st.warning("Harita oluşturulamadı.")
        with st.expander("Teknik hata detayı"):
            st.exception(e)


def render_record_detail(df: pd.DataFrame):
    section_header("🏥 Kayıt Detayı", "Seçilen barınak/bakımevi için operasyonel özet.")

    if df.empty:
        st.warning("Filtreye uygun kayıt bulunamadı.")
        return

    view = df.reset_index(drop=True).copy()
    name_str = view["name"].fillna("Bilinmeyen").astype(str)
    city_str = view["city"].fillna("—").astype(str)
    district_str = view["district"].fillna("—").astype(str)
    view["record_label"] = name_str + " · " + city_str + " / " + district_str + " · #" + view.index.astype(str)

    selected_label = st.selectbox("Kayıt seç", view["record_label"].tolist())
    row = view[view["record_label"] == selected_label].iloc[0]

    st.markdown("#### Temel Bilgiler")
    st.write(f"**Ad:** {row.get('name', '—')}")
    st.write(f"**İl:** {row.get('city', '—')}")
    st.write(f"**İlçe:** {row.get('district', '—')}")
    st.write(f"**Veri Kapsamı:** `{row.get('data_scope', '—')}`")

    if pd.notna(row.get("capacity")):
        st.markdown("#### Kapasite & Doluluk")
        c1, c2 = st.columns(2)
        c1.metric("Kapasite", int(row["capacity"]))
        if pd.notna(row.get("occupancy")):
            c2.metric("Mevcut Hayvan", int(row["occupancy"]))

    if pd.notna(row.get("risk_score")):
        st.markdown("#### Risk")
        st.metric("Risk Skoru", f"{row['risk_score']:.1f}", row.get("risk_level", "—"))


def render_dashboard_tabs(df, district_summary, history_df):
    section_header("📊 Dashboard", "Risk, doluluk, ilçe özeti ve raporlar.")

    tabs = st.tabs(["⚠️ Risk", "🏠 Doluluk", "🗺️ İlçe", "📈 Tarihsel", "📥 Rapor"])

    with tabs[0]:
        risk_df = df[as_bool_series(df["risk_eligible"])]
        if risk_df.empty:
            st.info("Risk analizine uygun kayıt yok.")
        else:
            try:
                fig = chart_risk_score(risk_df)
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.warning(f"Grafik hatası: {e}")

    with tabs[1]:
        occ_df = df[as_bool_series(df["occupancy_available"])]
        if occ_df.empty:
            st.info("Doluluk verisi yok.")
        else:
            try:
                fig = chart_occupancy_rate(occ_df)
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.warning(f"Grafik hatası: {e}")

    with tabs[2]:
        if district_summary is None or district_summary.empty:
            st.info("İlçe özeti hesaplanamadı.")
        else:
            st.dataframe(district_summary, width="stretch")
            try:
                fig = chart_district_avg_risk(district_summary)
                st.plotly_chart(fig, width="stretch")
            except Exception:
                pass

    with tabs[3]:
        if history_df is None or history_df.empty:
            st.info("Tarihsel snapshot bulunamadı.")
        else:
            st.dataframe(history_df.tail(20), width="stretch")

    with tabs[4]:
        st.download_button(
            "⬇️ Filtrelenmiş Veri (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="smartshelter_filtered.csv",
            mime="text/csv",
        )
        if district_summary is not None and not district_summary.empty:
            st.download_button(
                "⬇️ İlçe Özeti (CSV)",
                data=district_summary.to_csv(index=False).encode("utf-8"),
                file_name="smartshelter_district_summary.csv",
                mime="text/csv",
            )


# =========================================================
# MAIN
# =========================================================
def main():
    inject_css()
    render_hero()

    # Sidebar
    mode, rows_per_query, max_resources, strict_mode = render_sidebar()

    # Data pipeline (progress bar dahil)
    df, candidate_df, loaded_info, source_name, resource_label = load_pipeline(
        mode, rows_per_query, max_resources
    )

    st.success(f"✅ Aktif kaynak: **{source_name}** · {resource_label}")

    # Risk hesapla
    try:
        df = calculate_risk(df)
        df = create_action_recommendations(df)
    except Exception as e:
        st.warning(f"Risk hesaplama atlandı: {e}")

    df = ensure_app_columns(df)

    # Strict Mode filtre — gevşek versiyon
    if strict_mode and not df.empty:
        before = len(df)
        valid_scopes = ["risk_ready", "capacity_only", "location_only"]
        candidate = df[df["data_scope"].astype(str).isin(valid_scopes)].copy()
        candidate = candidate[as_bool_series(candidate["name_available"])]

        if not candidate.empty:
            df = candidate
            removed = before - len(df)
            if removed > 0:
                st.sidebar.success(f"🔒 Strict: {removed} kayıt çıkarıldı.")
        else:
            st.sidebar.warning(
                "⚠️ Strict Mode tüm kayıtları elerdi. İsim alanı dolu kayıtlar gösteriliyor."
            )
            relaxed = df[as_bool_series(df["name_available"])]
            df = relaxed if not relaxed.empty else df

    # ---- Sidebar filtreleri ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Filtreler")

    cities = sorted(df["city"].dropna().astype(str).unique().tolist()) if not df.empty else []
    districts = sorted(df["district"].dropna().astype(str).unique().tolist()) if not df.empty else []
    risk_levels = sorted(df["risk_level"].dropna().astype(str).unique().tolist()) if not df.empty else []

    use_city = st.sidebar.checkbox("İl filtresi kullan", value=False)
    selected_cities = (
        st.sidebar.multiselect("İl seç", cities, default=cities) if use_city else cities
    )

    use_district = st.sidebar.checkbox("İlçe filtresi kullan", value=False)
    selected_districts = (
        st.sidebar.multiselect("İlçe seç", districts, default=districts) if use_district else districts
    )

    # Akıllı default
    if risk_levels:
        default_risks = [r for r in risk_levels if r != "Veri yetersiz"]
        if not default_risks:
            default_risks = risk_levels
    else:
        default_risks = []

    selected_risks = st.sidebar.multiselect(
        "Risk seviyesi", risk_levels, default=default_risks,
    )

    if not selected_risks and risk_levels:
        st.sidebar.info("ℹ️ Risk filtresi boş — tüm seviyeler gösteriliyor.")
        selected_risks = risk_levels

    # ---- Filtre uygula ----
    if df.empty:
        filtered = df.copy()
    else:
        filtered = df[
            df["city"].astype(str).isin(selected_cities)
            & df["district"].astype(str).isin(selected_districts)
            & df["risk_level"].astype(str).isin(selected_risks)
        ].copy()

    # ---- Render ----
    render_kpis(filtered)
    st.divider()

    left, right = st.columns([2.25, 1], gap="large")
    with left:
        render_map(filtered)
    with right:
        render_record_detail(filtered)

    st.divider()

    try:
        district_summary = build_district_summary(filtered)
    except Exception:
        district_summary = pd.DataFrame()

    history_df = pd.read_csv(HISTORY_FILE) if HISTORY_FILE.exists() else pd.DataFrame()

    render_dashboard_tabs(filtered, district_summary, history_df)

    with st.expander("🧾 Ham Veri", expanded=False):
        st.dataframe(filtered, width="stretch")


if __name__ == "__main__":
    main()
