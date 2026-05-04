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
MAX_MAP_POINTS = 5000


# =========================================================
# CACHED LOADERS
# =========================================================
@st.cache_data(show_spinner=False)
def cached_load_local_data(path):
    return load_local_data(path)


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


# =========================================================
# ENSURE APP COLUMNS — vektörize, NA-safe
# =========================================================
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

    # ---- İSİM AKILLI TÜRETME (vektörize, NA-safe) ----
    name_alt_cols = [
        "adi", "adı", "ad", "tesis_adi", "tesis_adı",
        "kurum_adi", "kurum_adı", "merkez_adi", "merkez_adı",
        "title", "isim", "barinak_adi", "barınak_adı",
        "label", "mahalle", "mahalle_adi", "mahalle_adı",
        "sokak", "cadde",
    ]

    def clean_str_series(s):
        out = s.astype(str).str.strip()
        out = out.replace({"nan": "", "None": "", "<NA>": "", "NaN": ""})
        return out

    name_clean = clean_str_series(df["name"])

    for col in name_alt_cols:
        if col in df.columns:
            mask = name_clean.eq("")
            if mask.any():
                alt = clean_str_series(df[col])
                name_clean = name_clean.where(~mask, alt)

    mask = name_clean.eq("")
    if mask.any():
        district_s = clean_str_series(df["district"])
        city_s = clean_str_series(df["city"])
        portal_s = clean_str_series(df["source_portal"])

        def combine_loc(d, c, p):
            parts = [x for x in [d, c] if x]
            loc = " / ".join(parts)
            if loc and p:
                return f"{p} · {loc}"
            if loc:
                return loc
            if p:
                return f"{p} kaydı"
            return ""

        derived = pd.Series(
            [combine_loc(d, c, p) for d, c, p in zip(district_s, city_s, portal_s)],
            index=df.index,
        )
        name_clean = name_clean.where(~mask, derived)

    mask = name_clean.eq("")
    if mask.any():
        lat_s = pd.to_numeric(df["lat"], errors="coerce")
        lon_s = pd.to_numeric(df["lon"], errors="coerce")
        coord_label = pd.Series(
            [
                f"Kayıt @ {la:.3f}, {lo:.3f}" if pd.notna(la) and pd.notna(lo) else ""
                for la, lo in zip(lat_s, lon_s)
            ],
            index=df.index,
        )
        name_clean = name_clean.where(~mask, coord_label)

    df["name"] = name_clean

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

    df["name_available"] = df["name"].astype(str).str.strip().ne("")
    df["coordinate_valid"] = (
        df["lat"].between(-90, 90, inclusive="both")
        & df["lon"].between(-180, 180, inclusive="both")
    )
    df["capacity_available"] = df["capacity"].notna() & (df["capacity"] > 0)
    df["occupancy_available"] = df["occupancy"].notna()
    df["risk_eligible"] = df["capacity_available"] & df["occupancy_available"]

    df["data_scope"] = "metadata_only"
    df.loc[df["coordinate_valid"], "data_scope"] = "location_only"
    df.loc[df["risk_eligible"], "data_scope"] = "capacity_only"
    df.loc[df["risk_eligible"] & df["coordinate_valid"], "data_scope"] = "risk_ready"

    return df


# =========================================================
# CKAN TARAMA — PROGRESS BAR
# =========================================================
def search_turkiye_with_progress(rows_per_query):
    cache_key = f"ckan_search_cache_{rows_per_query}"
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        st.success(f"⚡ Önbellek kullanıldı · {len(cached)} resource (yeniden tarama yok)")
        return cached

    total_portals = len(TURKIYE_CKAN_SOURCES)

    st.markdown("#### 🇹🇷 Türkiye Geneli CKAN Taraması")
    info_box = st.empty()
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    info_box.info(f"🔄 {total_portals} portal paralel olarak taranıyor...")

    portal_log = []

    def on_portal_done(portal_name, current, total):
        try:
            ratio = min(max(current / max(total, 1), 0.0), 1.0)
            progress_bar.progress(ratio)
            portal_log.append(f"✅ {portal_name}")
            recent = portal_log[-5:]
            status_text.markdown(
                f"**{current}/{total}** portal tamamlandı · Son tamamlananlar:\n\n"
                + "\n".join(f"- {p}" for p in recent)
            )
        except Exception:
            pass

    try:
        result = search_turkiye_ckan_resources_with_progress(
            rows_per_query=rows_per_query,
            on_portal_done=on_portal_done,
        )
    except Exception as e:
        status_text.error(f"❌ Tarama hatası: {e}")
        result = []

    progress_bar.progress(1.0)

    if isinstance(result, pd.DataFrame):
        df = result.copy()
    elif isinstance(result, list):
        df = pd.DataFrame(result) if result else pd.DataFrame()
    else:
        df = pd.DataFrame()

    info_box.success(
        f"✅ Tarama tamamlandı: **{len(df)}** uygun resource bulundu "
        f"({total_portals} portal tarandı)"
    )
    status_text.empty()

    st.session_state[cache_key] = df
    return df


def load_resources_with_progress(candidate_records, max_resources):
    urls = tuple(sorted(r.get("url", "") for r in candidate_records[:max_resources]))
    cache_key = f"resources_loaded_{hash(urls)}_{max_resources}"

    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        st.success(f"⚡ Önbellek kullanıldı · {len(cached[1])} resource yüklü")
        return cached

    total = min(max_resources, len(candidate_records))

    st.markdown("#### 📥 Resource'lar İndiriliyor")
    info_box = st.empty()
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    info_box.info(f"🔄 {total} resource paralel olarak indiriliyor...")

    download_log = []

    def on_resource_done(resource, done, total_n):
        try:
            ratio = min(max(done / max(total_n, 1), 0.0), 1.0)
            progress_bar.progress(ratio)
            portal = resource.get("source_portal", "?")
            name = resource.get("name", "?")[:50]
            fmt = str(resource.get("format", "")).upper()
            download_log.append(f"📦 [{portal}] {name} ({fmt})")
            recent = download_log[-5:]
            status_text.markdown(
                f"**{done}/{total_n}** resource indirildi · Son tamamlananlar:\n\n"
                + "\n".join(f"- {p}" for p in recent)
            )
        except Exception:
            pass

    try:
        result = load_multiple_resources(
            candidate_records,
            max_resources=max_resources,
            on_resource_done=on_resource_done,
        )
        if isinstance(result, tuple) and len(result) == 3:
            df_loaded, loaded_info, failed = result
        elif isinstance(result, tuple) and len(result) == 2:
            df_loaded, loaded_info = result
            failed = []
        else:
            df_loaded = result if isinstance(result, pd.DataFrame) else pd.DataFrame()
            loaded_info = []
            failed = []
    except Exception as e:
        info_box.error(f"❌ Resource yükleme hatası: {e}")
        df_loaded = pd.DataFrame()
        loaded_info = []
        failed = []

    progress_bar.progress(1.0)
    info_box.success(
        f"✅ İndirme tamamlandı: **{len(loaded_info)}** resource yüklü, "
        f"**{len(failed)}** başarısız · **{len(df_loaded)}** satır"
    )
    status_text.empty()

    st.session_state[cache_key] = (df_loaded, loaded_info, failed)
    return df_loaded, loaded_info, failed


# =========================================================
# SIDEBAR
# =========================================================
def render_sidebar():
    st.sidebar.title("⚙️ Kontrol Paneli")

    if st.sidebar.button("🔄 Cache Temizle", width="stretch"):
        st.cache_data.clear()
        keys_to_remove = [
            k for k in st.session_state.keys()
            if k.startswith("ckan_search_cache_")
            or k.startswith("resources_loaded_")
            or k.startswith("filter_")
            or k.startswith("risk_filter_")
            or k.startswith("city_filter_")
            or k.startswith("district_filter_")
            or k.startswith("use_city_")
            or k.startswith("use_district_")
        ]
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
# DATA PIPELINE
# =========================================================
def load_pipeline(mode, rows_per_query, max_resources):
    candidate_df = pd.DataFrame()
    loaded_info = []

    demo_raw = cached_load_local_data(LOCAL_FILE).copy()
    demo_df = normalize_columns(demo_raw)
    demo_df = ensure_app_columns(demo_df)
    demo_df["source_portal"] = "Demo CSV"

    if mode == "Stabil Demo CSV":
        return demo_df, candidate_df, loaded_info, "Stabil Demo CSV", "data/kocaeli_shelters.csv"

    candidate_df = search_turkiye_with_progress(rows_per_query)

    if candidate_df.empty:
        st.warning("CKAN taramasında uygun resource bulunamadı. Demo veri kullanılıyor.")
        return demo_df, candidate_df, loaded_info, "Stabil Demo CSV (CKAN boş)", "data/kocaeli_shelters.csv"

    candidate_records = candidate_df.to_dict(orient="records")

    with st.expander("🇹🇷 Bulunan Resource Adayları", expanded=False):
        st.dataframe(candidate_df, width="stretch", height=300)

    df_loaded, loaded_info, failed = load_resources_with_progress(candidate_records, max_resources)

    if df_loaded.empty:
        st.warning("CKAN'dan veri yüklenemedi. Demo veri kullanılıyor.")
        return demo_df, candidate_df, loaded_info, "Stabil Demo CSV (CKAN okunamadı)", "data/kocaeli_shelters.csv"

    df_loaded = normalize_columns(df_loaded)
    df_loaded = ensure_app_columns(df_loaded)
    if "source_portal" not in df_loaded.columns or df_loaded["source_portal"].astype(str).eq("").all():
        df_loaded["source_portal"] = "CKAN"

    usable_ckan = int(df_loaded["coordinate_valid"].sum())
    has_cap_ckan = int(df_loaded["capacity_available"].sum())
    has_name_ckan = int(df_loaded["name_available"].sum())

    st.info(
        f"📊 CKAN'dan **{len(df_loaded):,}** satır yüklendi "
        f"({usable_ckan:,} koordinatlı, {has_cap_ckan:,} kapasite, {has_name_ckan:,} isim). "
        f"Demo verisi ({len(demo_df)} kayıt) ile birleştiriliyor."
    )

    combined = pd.concat([demo_df, df_loaded], ignore_index=True)
    combined = ensure_app_columns(combined)

    return (
        combined,
        candidate_df,
        loaded_info,
        "Hibrit (Demo + CKAN)",
        f"Demo + {len(loaded_info)} CKAN resource",
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
    c1.metric("Envanter Kaydı", f"{len(df):,}")
    c2.metric("Risk Hazır", f"{int(risk_ok.sum()):,}")
    c3.metric("Bilinen Kapasite", f"{known_cap:,}")
    c4.metric("Bilinen Mevcut", f"{known_occ:,}")
    c5.metric("Ortalama Risk", f"{avg_risk:.1f}" if avg_risk is not None else "—")
    c6.metric("Geçerli Koordinat", f"{int(coord_ok.sum()):,}")

    if critical > 0:
        st.error(f"🚨 {critical:,} kritik kayıt mevcut.")
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
            st.write(f"Toplam kayıt: {len(df):,}")
            st.write(f"Geçerli koordinatlı: {int(as_bool_series(df['coordinate_valid']).sum()):,}")
            st.write(f"İsim alanı dolu: {int(as_bool_series(df['name_available']).sum()):,}")
            if not df.empty:
                cols = [c for c in ["name", "city", "district", "lat", "lon"] if c in df.columns]
                if cols:
                    st.dataframe(df[cols].head(10), width="stretch")
        return

    original_count = len(map_df)
    sampled = False
    if original_count > MAX_MAP_POINTS:
        priority = map_df[as_bool_series(map_df["risk_eligible"])]
        rest = map_df[~as_bool_series(map_df["risk_eligible"])]

        if len(priority) >= MAX_MAP_POINTS:
            map_df = priority.sample(n=MAX_MAP_POINTS, random_state=42)
        else:
            remaining = MAX_MAP_POINTS - len(priority)
            map_df = pd.concat([
                priority,
                rest.sample(n=min(remaining, len(rest)), random_state=42)
            ], ignore_index=True)
        sampled = True

    if sampled:
        st.warning(
            f"⚡ Performans için {original_count:,} kayıttan {len(map_df):,} tanesi haritada gösteriliyor "
            f"(risk-uygun olanlar öncelikli)."
        )

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

    view_source = df
    if len(df) > 500:
        priority = df[as_bool_series(df["risk_eligible"]) | as_bool_series(df["capacity_available"])]
        if not priority.empty:
            view_source = priority.head(500)
            st.caption(f"ℹ️ {len(df):,} kayıttan ilk 500 öncelikli kayıt listeleniyor.")
        else:
            view_source = df.head(500)
            st.caption(f"ℹ️ {len(df):,} kayıttan ilk 500 listeleniyor.")

    view = view_source.reset_index(drop=True).copy()
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
    st.write(f"**Kaynak Portal:** {row.get('source_portal', '—')}")

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

    mode, rows_per_query, max_resources, strict_mode = render_sidebar()

    df, candidate_df, loaded_info, source_name, resource_label = load_pipeline(
        mode, rows_per_query, max_resources
    )

    st.success(f"✅ Aktif kaynak: **{source_name}** · {resource_label}")

    try:
        df = calculate_risk(df)
        df = create_action_recommendations(df)
    except Exception as e:
        st.warning(f"Risk hesaplama atlandı: {e}")

    df = ensure_app_columns(df)

    # ---- AKILLI STRICT MODE ----
    if strict_mode and not df.empty:
        before = len(df)
        valid_scopes = ["risk_ready", "capacity_only", "location_only"]

        candidate = df[df["data_scope"].astype(str).isin(valid_scopes)].copy()
        with_name = candidate[as_bool_series(candidate["name_available"])]

        if not with_name.empty:
            df = with_name
            removed = before - len(df)
            if removed > 0:
                st.sidebar.success(f"🔒 Strict: {removed:,} kayıt çıkarıldı.")
                st.sidebar.caption(
                    f"📊 Kalan: {len(df):,} kayıt · "
                    f"{int(as_bool_series(df['coordinate_valid']).sum()):,} koordinatlı"
                )
        elif not candidate.empty:
            df = candidate
            st.sidebar.warning(
                f"⚠️ Strict: name filtresi atlandı. "
                f"{len(df):,} scope-uygun kayıt gösteriliyor."
            )
        else:
            relaxed = df[as_bool_series(df["coordinate_valid"]) | as_bool_series(df["name_available"])]
            df = relaxed if not relaxed.empty else df
            st.sidebar.warning("⚠️ Strict Mode tüm filtreleri yedi. Gevşek mod aktif.")

    # ---- Sidebar filtreleri ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Filtreler")

    # 🌍 Tüm Türkiye Görünümü kısayolu — tüm filtreleri sıfırlar
    if st.sidebar.button(
        "🌍 Tüm Türkiye Görünümü",
        width="stretch",
        help="Tüm filtreleri sıfırlar, tüm kayıtları gösterir.",
    ):
        keys_to_remove = [
            k for k in st.session_state.keys()
            if k.startswith("filter_")
            or k.startswith("risk_filter_")
            or k.startswith("city_filter_")
            or k.startswith("district_filter_")
            or k.startswith("use_city_")
            or k.startswith("use_district_")
        ]
        for k in keys_to_remove:
            del st.session_state[k]
        st.rerun()

    cities = sorted(df["city"].dropna().astype(str).unique().tolist()) if not df.empty else []
    districts = sorted(df["district"].dropna().astype(str).unique().tolist()) if not df.empty else []
    risk_levels = sorted(df["risk_level"].dropna().astype(str).unique().tolist()) if not df.empty else []

    # Mode-bağımlı key — mode değişince filtreler resetlenir
    filter_key = f"{mode}_{len(df)}"

    use_city = st.sidebar.checkbox(
        "İl filtresi kullan", value=False, key=f"use_city_{filter_key}"
    )
    selected_cities = (
        st.sidebar.multiselect(
            "İl seç", cities, default=cities,
            key=f"city_filter_{filter_key}",
        ) if use_city else cities
    )

    use_district = st.sidebar.checkbox(
        "İlçe filtresi kullan", value=False, key=f"use_district_{filter_key}"
    )
    selected_districts = (
        st.sidebar.multiselect(
            "İlçe seç", districts, default=districts,
            key=f"district_filter_{filter_key}",
        ) if use_district else districts
    )

    # ---- AKILLI RİSK FİLTRESİ ----
    if risk_levels and not df.empty:
        veri_yetersiz_count = (df["risk_level"].astype(str) == "Veri yetersiz").sum()
        total_count = len(df)
        veri_yetersiz_ratio = veri_yetersiz_count / max(total_count, 1)

        if veri_yetersiz_ratio > 0.5:
            default_risks = risk_levels
            st.sidebar.caption(
                f"ℹ️ Veri kapsamı geniş — tüm risk seviyeleri varsayılan olarak seçili "
                f"({veri_yetersiz_count:,} kayıt 'Veri yetersiz')."
            )
        else:
            default_risks = [r for r in risk_levels if r != "Veri yetersiz"]
            if not default_risks:
                default_risks = risk_levels
    else:
        default_risks = risk_levels if risk_levels else []

    selected_risks = st.sidebar.multiselect(
        "Risk seviyesi", risk_levels, default=default_risks,
        key=f"risk_filter_{filter_key}",
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

    # ---- FİLTRE TEŞHİS PANELİ ----
    with st.expander(
        f"🔬 Filtre Teşhisi (Toplam: {len(df):,} → Filtrelenmiş: {len(filtered):,})",
        expanded=False,
    ):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Strict Mode Sonrası:**")
            st.write(f"- Toplam kayıt: **{len(df):,}**")
            st.write(f"- Şehir sayısı: {len(cities)}")
            st.write(f"- İlçe sayısı: {len(districts)}")
            st.write(f"- Koordinatlı: {int(as_bool_series(df['coordinate_valid']).sum()):,}")
        with col2:
            st.markdown("**Aktif Filtreler:**")
            st.write(f"- İl: {'✓ aktif' if use_city else '✗ kapalı (hepsi seçili)'}")
            st.write(f"- İlçe: {'✓ aktif' if use_district else '✗ kapalı (hepsi seçili)'}")
            st.write(f"- Risk: **{len(selected_risks)}/{len(risk_levels)}** seçili")
            if len(selected_risks) < len(risk_levels):
                missing = set(risk_levels) - set(selected_risks)
                st.caption(f"❗ Filtrelenen: {', '.join(missing)}")
        with col3:
            st.markdown("**Risk Seviyeleri (df):**")
            if not df.empty:
                risk_counts = df["risk_level"].astype(str).value_counts()
                for level, count in risk_counts.items():
                    check = "✓" if level in selected_risks else "✗"
                    st.write(f"- {check} {level}: **{count:,}**")

        if len(filtered) < len(df):
            elenen = len(df) - len(filtered)
            st.error(
                f"⚠️ **{elenen:,} kayıt filtrelendi.** "
                f"Tümünü görmek için yan paneldeki 🌍 **Tüm Türkiye Görünümü** butonuna basın."
            )

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
        if len(filtered) > 1000:
            st.caption(f"ℹ️ {len(filtered):,} kayıttan ilk 1000 gösteriliyor.")
            st.dataframe(filtered.head(1000), width="stretch")
        else:
            st.dataframe(filtered, width="stretch")


if __name__ == "__main__":
    main()
