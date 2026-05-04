from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# =============================================================================
# Path / import hazırlığı
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

# Sende dosya yapısı büyük ihtimalle:
# app.py
# src/
#   src/
#     map.py
#     data_loader.py
#
# Aşağıdaki path eklemeleri farklı import senaryolarını tolere eder.
CANDIDATE_PATHS = [
    ROOT_DIR,
    ROOT_DIR / "src",
    ROOT_DIR / "src" / "src",
]

for p in CANDIDATE_PATHS:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _import_project_modules():
    """
    Projedeki data_loader ve map modüllerini farklı path yapılarında bulmaya çalışır.
    """
    load_shelter_dataset = None
    create_shelter_map = None

    loader_errors = []
    map_errors = []

    # data_loader import denemeleri
    loader_imports = [
        "src.data_loader",
        "src.src.data_loader",
        "data_loader",
    ]

    for module_name in loader_imports:
        try:
            module = __import__(module_name, fromlist=["load_shelter_dataset"])
            load_shelter_dataset = getattr(module, "load_shelter_dataset")
            break
        except Exception as e:
            loader_errors.append(f"{module_name}: {e}")

    # map import denemeleri
    map_imports = [
        "src.map",
        "src.src.map",
        "map",
    ]

    for module_name in map_imports:
        try:
            module = __import__(module_name, fromlist=["create_shelter_map"])
            create_shelter_map = getattr(module, "create_shelter_map")
            break
        except Exception as e:
            map_errors.append(f"{module_name}: {e}")

    if load_shelter_dataset is None:
        st.error("`load_shelter_dataset` import edilemedi.")
        with st.expander("Import hata detayları", expanded=True):
            st.write(loader_errors)
            st.write("sys.path:", sys.path)
        st.stop()

    if create_shelter_map is None:
        st.error("`create_shelter_map` import edilemedi.")
        with st.expander("Import hata detayları", expanded=True):
            st.write(map_errors)
            st.write("sys.path:", sys.path)
        st.stop()

    return load_shelter_dataset, create_shelter_map


load_shelter_dataset, create_shelter_map = _import_project_modules()


try:
    from streamlit_folium import st_folium
except Exception:
    st_folium = None


# =============================================================================
# Streamlit ayarları
# =============================================================================

st.set_page_config(
    page_title="SmartShelter GIS",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS
# =============================================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background: #f8fafc;
        }

        .main-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #172554 100%);
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.2rem;
        }

        .main-header h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
        }

        .main-header p {
            margin: 0.45rem 0 0 0;
            color: #dbeafe;
            font-size: 0.98rem;
        }

        .metric-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        }

        .metric-label {
            color: #64748b;
            font-size: 0.88rem;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            color: #0f172a;
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .section-title {
            margin-top: 1rem;
            margin-bottom: 0.25rem;
            font-size: 1.45rem;
            font-weight: 800;
            color: #0f172a;
        }

        .section-subtitle {
            color: #94a3b8;
            margin-bottom: 1rem;
        }

        .source-ok {
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            color: #166534;
            padding: 0.9rem 1rem;
            border-radius: 12px;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        .source-warn {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #9a3412;
            padding: 0.9rem 1rem;
            border-radius: 12px;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        .source-error {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
            padding: 0.9rem 1rem;
            border-radius: 12px;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 2rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.5rem 1rem;
            background: #f1f5f9;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Yardımcı fonksiyonlar
# =============================================================================

def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _format_int(value: Any) -> str:
    try:
        if pd.isna(value):
            return "0"
        return f"{int(value):,}"
    except Exception:
        return "0"


def _format_float(value: Any, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "—"


def _column_exists(df: pd.DataFrame, col: str) -> bool:
    return df is not None and not df.empty and col in df.columns


def _valid_coordinate_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0

    lat_col = "lat" if "lat" in df.columns else "latitude" if "latitude" in df.columns else None
    lon_col = "lon" if "lon" in df.columns else "longitude" if "longitude" in df.columns else None

    if not lat_col or not lon_col:
        return 0

    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    valid = (
        lat.notna()
        & lon.notna()
        & lat.between(-90, 90)
        & lon.between(-180, 180)
    )

    return int(valid.sum())


def _risk_order(values: list[str]) -> list[str]:
    order = ["Düşük", "Orta", "Yüksek", "Kritik", "Veri yetersiz"]
    known = [x for x in order if x in values]
    unknown = sorted([x for x in values if x not in order])
    return known + unknown


def _clear_filter_state():
    keys = [
        "city_filter_enabled_v4",
        "district_filter_enabled_v4",
        "risk_filter_v4",
        "city_filter_v4",
        "district_filter_v4",
        "search_text_v4",
    ]

    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_load_data(mode: str, strict_mode: bool, cache_buster: int):
    """
    CKAN sorgularını ve demo CSV okumasını cache'ler.

    cache_buster, kullanıcı 'Veriyi yeniden çek' dediğinde değişir.
    """
    return load_shelter_dataset(
        mode=mode,
        strict_mode=strict_mode,
    )


def _apply_dashboard_filters(
    df: pd.DataFrame,
    selected_cities: list[str] | None,
    selected_districts: list[str] | None,
    selected_risks: list[str] | None,
    search_text: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if selected_cities and "city" in out.columns:
        out = out[out["city"].astype(str).isin(selected_cities)]

    if selected_districts and "district" in out.columns:
        out = out[out["district"].astype(str).isin(selected_districts)]

    if selected_risks and "risk_level" in out.columns:
        out = out[out["risk_level"].astype(str).isin(selected_risks)]

    if search_text and "name" in out.columns:
        q = search_text.strip().lower()
        if q:
            out = out[out["name"].astype(str).str.lower().str.contains(q, na=False)]

    return out.reset_index(drop=True)


def _render_metric_cards(df: pd.DataFrame):
    total_records = len(df) if df is not None else 0

    risk_ready = 0
    if _column_exists(df, "risk_score"):
        risk_ready = int(pd.to_numeric(df["risk_score"], errors="coerce").notna().sum())

    known_capacity = 0
    if _column_exists(df, "capacity"):
        known_capacity = pd.to_numeric(df["capacity"], errors="coerce").sum(skipna=True)

    known_occupancy = 0
    if _column_exists(df, "occupancy"):
        known_occupancy = pd.to_numeric(df["occupancy"], errors="coerce").sum(skipna=True)

    avg_risk = None
    if _column_exists(df, "risk_score"):
        avg_risk = pd.to_numeric(df["risk_score"], errors="coerce").mean()

    valid_coords = _valid_coordinate_count(df)

    cols = st.columns(6)

    with cols[0]:
        st.metric("Envanter Kaydı", _format_int(total_records))

    with cols[1]:
        st.metric("Risk Hazır", _format_int(risk_ready))

    with cols[2]:
        st.metric("Bilinen Kapasite", _format_int(known_capacity))

    with cols[3]:
        st.metric("Bilinen Mevcut", _format_int(known_occupancy))

    with cols[4]:
        st.metric("Ortalama Risk", _format_float(avg_risk, 1))

    with cols[5]:
        st.metric("Geçerli Koordinat", _format_int(valid_coords))


def _render_filter_diagnostics(
    raw_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    selected_cities: list[str] | None,
    selected_districts: list[str] | None,
    selected_risks: list[str] | None,
):
    total = len(raw_df) if raw_df is not None else 0
    shown = len(filtered_df) if filtered_df is not None else 0

    cities_count = raw_df["city"].nunique() if _column_exists(raw_df, "city") else 0
    districts_count = raw_df["district"].nunique() if _column_exists(raw_df, "district") else 0
    coord_count = _valid_coordinate_count(raw_df)

    risk_counts = {}
    if _column_exists(raw_df, "risk_level"):
        risk_counts = raw_df["risk_level"].astype(str).value_counts().to_dict()

    with st.expander(f"🔬 Filtre Teşhisi (Toplam: {total} → Görünen: {shown})", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Strict Mode Sonrası / Ham Veri:**")
            st.write(f"- Toplam: **{total}**")
            st.write(f"- Şehir: **{cities_count}**")
            st.write(f"- İlçe: **{districts_count}**")
            st.write(f"- Koordinatlı: **{coord_count}**")

        with c2:
            st.markdown("**Aktif Filtreler:**")
            st.write(f"- İl: **{len(selected_cities) if selected_cities else 0} seçili**")
            st.write(f"- İlçe: **{len(selected_districts) if selected_districts else 0} seçili**")
            st.write(f"- Risk: **{len(selected_risks) if selected_risks else 0} seçili**")

        with c3:
            st.markdown("**Risk Seviyeleri:**")
            if risk_counts:
                for risk, count in risk_counts.items():
                    st.write(f"- {risk}: **{count}**")
            else:
                st.write("- Risk verisi yok")


def _render_map(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("Haritada gösterilecek kayıt yok.")
        return

    coord_count = _valid_coordinate_count(df)

    if coord_count == 0:
        st.warning(
            "Bu filtrelerde geçerli koordinatı olan kayıt yok. "
            "Dashboard kayıtları görünebilir ama haritada marker oluşmaz."
        )

    m = create_shelter_map(
        df,
        default_tile="Stadia OSM Bright",
    )

    if st_folium is not None:
        st_folium(
            m,
            width=None,
            height=680,
            returned_objects=[],
            key="main_shelter_map_v4",
        )
    else:
        st.warning(
            "`streamlit-folium` paketi bulunamadı. "
            "Harita HTML fallback ile gösteriliyor. "
            "requirements.txt içine `streamlit-folium` eklemen önerilir."
        )
        import streamlit.components.v1 as components

        components.html(
            m.get_root().render(),
            height=700,
            scrolling=False,
        )


def _render_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("Tabloda gösterilecek kayıt yok.")
        return

    preferred_cols = [
        "name",
        "city",
        "district",
        "capacity",
        "occupancy",
        "risk_score",
        "risk_level",
        "lat",
        "lon",
        "source_portal",
        "source_dataset",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    rest = [c for c in df.columns if c not in cols]
    show_df = df[cols + rest].copy()

    st.dataframe(
        show_df,
        use_container_width=True,
        height=480,
    )

    csv = show_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "⬇️ CSV indir",
        data=csv,
        file_name="smartshelter_filtered_data.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _render_risk_summary(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("Özet üretilecek kayıt yok.")
        return

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Risk Dağılımı")

        if "risk_level" in df.columns:
            risk_counts = (
                df["risk_level"]
                .astype(str)
                .value_counts()
                .rename_axis("Risk Seviyesi")
                .reset_index(name="Kayıt")
            )

            st.bar_chart(
                risk_counts.set_index("Risk Seviyesi"),
                use_container_width=True,
            )
            st.dataframe(risk_counts, use_container_width=True)
        else:
            st.info("risk_level kolonu yok.")

    with c2:
        st.subheader("Şehir Dağılımı")

        if "city" in df.columns:
            city_counts = (
                df["city"]
                .astype(str)
                .replace({"": "Bilinmiyor", "nan": "Bilinmiyor"})
                .value_counts()
                .head(20)
                .rename_axis("Şehir")
                .reset_index(name="Kayıt")
            )

            st.bar_chart(
                city_counts.set_index("Şehir"),
                use_container_width=True,
            )
            st.dataframe(city_counts, use_container_width=True)
        else:
            st.info("city kolonu yok.")


# =============================================================================
# Header
# =============================================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🐾 SmartShelter GIS Dashboard</h1>
        <p>
            Hayvan barınakları için kapasite, risk, konum ve veri kaynağı izleme paneli.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Sidebar - kaynak seçimi
# =============================================================================

with st.sidebar:
    st.header("Veri Kaynağı")

    data_mode = st.radio(
        "Veri modu",
        [
            "Stabil Demo CSV",
            "Türkiye Geneli CKAN Taraması",
        ],
        index=0,
        key="data_mode_v4",
    )

    st.divider()

    st.subheader("🔒 Veri Kalitesi Modu")

    strict_mode = st.toggle(
        "Strict Mode (Sadece Gerçek Veri)",
        value=False,
        help=(
            "Açıksa sadece koordinatı/kaynağı geçerli kayıtları tutar. "
            "Kapalıyken eksik CKAN kayıtları dashboardda kalabilir; "
            "koordinatsız olanlar sadece haritada görünmez."
        ),
        key="strict_mode_v4",
    )

    st.caption(
        "Not: CKAN seçiliyken veri alınamazsa demo CSV'ye otomatik dönülmez; "
        "hata/debug ekranda gösterilir."
    )

    st.divider()

    if "cache_buster_v4" not in st.session_state:
        st.session_state["cache_buster_v4"] = 0

    if st.button("🔄 Veriyi yeniden çek", use_container_width=True):
        st.session_state["cache_buster_v4"] += 1
        st.cache_data.clear()
        st.rerun()

    if st.button("🧹 Filtreleri sıfırla", use_container_width=True):
        _clear_filter_state()
        st.rerun()


# =============================================================================
# Veri yükleme
# =============================================================================

with st.spinner("Veri yükleniyor..."):
    data_result = _cached_load_data(
        mode=data_mode,
        strict_mode=strict_mode,
        cache_buster=st.session_state.get("cache_buster_v4", 0),
    )

raw_df = data_result.df if hasattr(data_result, "df") else pd.DataFrame()

source_label = getattr(data_result, "source_label", "Bilinmeyen kaynak")
is_demo = bool(getattr(data_result, "is_demo", False))
errors = getattr(data_result, "errors", []) or []
debug = getattr(data_result, "debug", {}) or {}


# =============================================================================
# Kaynak durumu
# =============================================================================

if errors and not is_demo:
    st.markdown(
        f"""
        <div class="source-error">
            ❌ Aktif kaynak: {source_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("CKAN hata / debug detayları", expanded=True):
        st.write("Hatalar:")
        st.write(errors)
        st.write("Debug:")
        st.write(debug)

elif errors and is_demo:
    st.markdown(
        f"""
        <div class="source-warn">
            ⚠️ Aktif kaynak: {source_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Demo veri hata / debug detayları", expanded=False):
        st.write(errors)
        st.write(debug)

else:
    st.markdown(
        f"""
        <div class="source-ok">
            ✅ Aktif kaynak: {source_label}
        </div>
        """,
        unsafe_allow_html=True,
    )


if raw_df is None or raw_df.empty:
    st.warning(
        "Görüntülenecek kayıt yok. "
        "Eğer CKAN modundaysan, debug detaylarında hangi portal/kaynakların denendiğini kontrol et."
    )

    with st.expander("Boş veri debug bilgisi", expanded=True):
        st.write("data_mode:", data_mode)
        st.write("strict_mode:", strict_mode)
        st.write("source_label:", source_label)
        st.write("errors:", errors)
        st.write("debug:", debug)

    st.stop()


# =============================================================================
# Sidebar - filtreler
# =============================================================================

with st.sidebar:
    st.header("🔎 Filtreler")

    search_text = st.text_input(
        "Barınak / tesis adı ara",
        value="",
        key="search_text_v4",
    )

    city_filter_enabled = st.checkbox(
        "İl filtresi kullan",
        value=False,
        key="city_filter_enabled_v4",
    )

    selected_cities: list[str] = []

    if city_filter_enabled and "city" in raw_df.columns:
        city_options = (
            raw_df["city"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .sort_values()
            .unique()
            .tolist()
        )

        selected_cities = st.multiselect(
            "İl seç",
            options=city_options,
            default=[],
            key="city_filter_v4",
        )

    district_filter_enabled = st.checkbox(
        "İlçe filtresi kullan",
        value=False,
        key="district_filter_enabled_v4",
    )

    selected_districts: list[str] = []

    district_base_df = raw_df.copy()

    if selected_cities and "city" in district_base_df.columns:
        district_base_df = district_base_df[
            district_base_df["city"].astype(str).isin(selected_cities)
        ]

    if district_filter_enabled and "district" in district_base_df.columns:
        district_options = (
            district_base_df["district"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .sort_values()
            .unique()
            .tolist()
        )

        selected_districts = st.multiselect(
            "İlçe seç",
            options=district_options,
            default=[],
            key="district_filter_v4",
        )

    st.caption("Risk seviyesi varsayılan: tümü")

    selected_risks: list[str] = []

    if "risk_level" in raw_df.columns:
        risk_options = (
            raw_df["risk_level"]
            .fillna("Veri yetersiz")
            .astype(str)
            .str.strip()
            .replace("", "Veri yetersiz")
            .unique()
            .tolist()
        )

        risk_options = _risk_order(risk_options)

        selected_risks = st.multiselect(
            "Risk seviyesi",
            options=risk_options,
            default=risk_options,
            key="risk_filter_v4",
        )

    else:
        st.info("risk_level kolonu bulunamadı.")


filtered_df = _apply_dashboard_filters(
    df=raw_df,
    selected_cities=selected_cities,
    selected_districts=selected_districts,
    selected_risks=selected_risks,
    search_text=search_text,
)


# =============================================================================
# Dashboard ana alanı
# =============================================================================

_render_filter_diagnostics(
    raw_df=raw_df,
    filtered_df=filtered_df,
    selected_cities=selected_cities,
    selected_districts=selected_districts,
    selected_risks=selected_risks,
)

st.markdown('<div class="section-title">📌 Anlık Durum</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Seçili filtrelere göre kapasite, risk ve veri kapsamı özeti.</div>',
    unsafe_allow_html=True,
)

_render_metric_cards(filtered_df)

st.divider()

tab_map, tab_summary, tab_data, tab_debug = st.tabs(
    [
        "🗺️ Harita",
        "📊 Özet",
        "📋 Veri Tablosu",
        "🛠️ Debug",
    ]
)

with tab_map:
    st.markdown("### 🗺️ Barınak Haritası")
    st.caption(
        "Sağ üstteki layer menüsünden Stadia, OpenStreetMap, CartoDB ve Esri temalarını seçebilirsin."
    )
    _render_map(filtered_df)

with tab_summary:
    st.markdown("### 📊 Veri Özeti")
    _render_risk_summary(filtered_df)

with tab_data:
    st.markdown("### 📋 Filtrelenmiş Veri")
    _render_table(filtered_df)

with tab_debug:
    st.markdown("### 🛠️ Debug Bilgileri")

    c1, c2 = st.columns(2)

    with c1:
        st.write("**Uygulama durumu**")
        st.json(
            {
                "data_mode": data_mode,
                "strict_mode": strict_mode,
                "source_label": source_label,
                "is_demo": is_demo,
                "raw_shape": list(raw_df.shape),
                "filtered_shape": list(filtered_df.shape),
                "valid_coordinates_raw": _valid_coordinate_count(raw_df),
                "valid_coordinates_filtered": _valid_coordinate_count(filtered_df),
            }
        )

    with c2:
        st.write("**Data loader debug**")
        st.write(debug)

    if errors:
        st.write("**Errors**")
        st.write(errors)

    st.write("**Kolonlar**")
    st.write(list(raw_df.columns))

    st.write("**İlk 20 kayıt**")
    st.dataframe(raw_df.head(20), use_container_width=True)
