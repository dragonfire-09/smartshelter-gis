from io import BytesIO

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

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
    load_local_data,
    load_resource,
    search_ckan_resources,
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


st.set_page_config(
    page_title="SmartShelter GIS",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }

        .small-note {
            color: #6b7280;
            font-size: 0.9rem;
        }

        .info-card {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }

        .warning-card {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }

        .danger-card {
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def to_excel_bytes(df: pd.DataFrame, district_summary: pd.DataFrame) -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Kayitlar")
        district_summary.to_excel(writer, index=False, sheet_name="Ilce_Ozeti")

    return output.getvalue()


def render_header():
    col_logo, col_title = st.columns([0.08, 0.92])

    with col_logo:
        st.markdown("## 🐾")

    with col_title:
        st.title("SmartShelter GIS")
        st.caption(
            "Hayvan bakımevleri için açık veri tabanlı GIS karar destek prototipi"
        )


def render_methodology_note():
    st.info(
        """
        Bu uygulama resmi bir denetim sistemi değildir. Açık veri ve demo veriyle çalışan
        prototip bir karar destek ekranıdır. Risk skoru; kapasite baskısı, veteriner yükü,
        sahiplendirme açığı ve kısırlaştırma performansı gibi göstergelerden hesaplanan
        örnek bir önceliklendirme skorudur.

        Geçmiş analitik modülü, uygulamanın çektiği verileri günlük snapshot olarak saklar.
        Kaynak sistem eski tarihli resource yayınlıyorsa derin CKAN taramasıyla listelenebilir.
        Kaynak yalnızca güncel dosya yayınlıyorsa, uygulama bundan sonraki her çekimi kendi
        tarihsel arşivine ekler.
        """
    )


def render_kpis(df: pd.DataFrame):
    total_records = len(df)
    total_capacity = int(df["capacity"].sum()) if total_records else 0
    total_occupancy = int(df["occupancy"].sum()) if total_records else 0
    avg_risk = df["risk_score"].mean() if total_records else 0
    critical_count = len(df[df["risk_level"].astype(str) == "Kritik"])

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Toplam Kayıt", total_records)
    col2.metric("Toplam Kapasite", total_capacity)
    col3.metric("Mevcut Hayvan", total_occupancy)
    col4.metric("Ortalama Risk", f"{avg_risk:.1f}" if total_records else "0")
    col5.metric("Kritik Kayıt", critical_count)

    if critical_count > 0:
        st.error(f"🔴 {critical_count} kayıt kritik risk seviyesinde.")
    else:
        st.success("🟢 Kritik seviyede kayıt bulunmuyor.")


def render_data_quality_summary(df: pd.DataFrame):
    total = len(df)

    if total == 0:
        st.warning("Veri bulunamadı.")
        return

    coord_missing = len(df[df["coordinate_valid"] == False])  # noqa: E712
    estimated = len(df[df["is_estimated"] == True])  # noqa: E712
    capacity_estimated = len(df[df["capacity_estimated"] == True])  # noqa: E712
    occupancy_estimated = len(df[df["occupancy_estimated"] == True])  # noqa: E712
    vet_estimated = len(df[df["vet_count_estimated"] == True])  # noqa: E712

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Toplam Kayıt", total)
    c2.metric("Geçersiz Koordinat", coord_missing)
    c3.metric("Tahmini Alan İçeren", estimated)
    c4.metric("Kapasite Tahmini", capacity_estimated)
    c5.metric("Veteriner Tahmini", vet_estimated)

    if estimated > 0:
        st.warning(
            "Bazı kayıtlarda eksik alanlar demo/prototip amacıyla tahmini değerlerle tamamlanmıştır."
        )
    else:
        st.success("Tahmini değer kullanılan kayıt bulunmuyor.")


def render_record_detail(df: pd.DataFrame):
    st.subheader("🏥 Kayıt Detayı")

    if len(df) == 0:
        st.warning("Filtreye uygun kayıt bulunamadı.")
        return

    names = df["name"].astype(str).tolist()

    selected_name = st.selectbox(
        "Kayıt seç",
        names,
        key="record_detail_select",
    )

    item = df[df["name"].astype(str) == selected_name].iloc[0]

    st.write(f"**İl:** {item['city']}")
    st.write(f"**İlçe:** {item['district']}")
    st.write(f"**Kapasite:** {int(item['capacity'])}")
    st.write(f"**Mevcut Hayvan:** {int(item['occupancy'])}")
    st.write(f"**Doluluk Oranı:** %{item['occupancy_rate']}")
    st.write(f"**Veteriner Sayısı:** {int(item['vet_count'])}")
    st.write(f"**Veteriner Başına Hayvan:** {item['animals_per_vet']:.1f}")
    st.write(f"**Kısırlaştırma:** {int(item['sterilization_count'])}")
    st.write(f"**Sahiplendirme:** {int(item['adoption_count'])}")
    st.write(f"**Risk Skoru:** {item['risk_score']}")

    risk_level = str(item["risk_level"])

    if risk_level == "Kritik":
        st.error("🔴 Kritik risk seviyesi")
    elif risk_level == "Orta":
        st.warning("🟠 Orta risk seviyesi")
    else:
        st.success("🟢 Düşük risk seviyesi")

    st.markdown("#### Önerilen Aksiyon")
    st.write(item["recommended_action"])

    if item.get("is_estimated", False):
        st.markdown("#### Veri Kalitesi Notu")
        st.warning(item.get("data_quality_note", "Bazı alanlar tahmini olabilir."))


def render_report_downloads(
    df: pd.DataFrame,
    district_summary: pd.DataFrame,
    history_df: pd.DataFrame,
    history_summary_df: pd.DataFrame,
):
    st.subheader("📥 Güncel Rapor İndirme")

    if len(df) == 0:
        st.warning("İndirilecek güncel kayıt bulunamadı.")
    else:
        csv_data = df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Güncel Filtrelenmiş CSV Rapor İndir",
            data=csv_data,
            file_name="smartshelter_filtered_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        excel_data = to_excel_bytes(df, district_summary)

        st.download_button(
            label="Güncel Excel Rapor İndir",
            data=excel_data,
            file_name="smartshelter_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("🕒 Tarihsel Veri İndirme")

    if history_df.empty:
        st.warning("Tarihsel snapshot verisi bulunmuyor.")
    else:
        history_csv = history_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Tüm Tarihsel Snapshot Verisini CSV İndir",
            data=history_csv,
            file_name="smartshelter_history_snapshots.csv",
            mime="text/csv",
            use_container_width=True,
        )

        history_summary_csv = history_summary_df.to_csv(index=False).encode(
            "utf-8-sig"
        )

        st.download_button(
            label="Tarihsel Özet CSV İndir",
            data=history_summary_csv,
            file_name="smartshelter_history_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_history_analytics(history_df: pd.DataFrame, history_summary_df: pd.DataFrame):
    st.subheader("🕒 Geçmiş Analitik ve Tarihsel Karşılaştırma")

    st.markdown(
        """
        Bu bölüm, uygulamanın daha önce çektiği verileri günlük snapshot olarak saklar.
        Böylece zaman içinde kapasite, mevcut hayvan sayısı, risk skoru ve kritik kayıt sayısı
        karşılaştırılabilir.
        """
    )

    available_dates = get_available_snapshot_dates(history_df)

    if len(available_dates) == 0:
        st.warning("Henüz tarihsel kayıt bulunmuyor.")
        return

    if len(available_dates) == 1:
        st.info(
            f"Şu anda yalnızca bir snapshot tarihi var: {available_dates[0]}. "
            "Karşılaştırma için farklı günlerde tekrar veri çekilmesi gerekir."
        )

        st.plotly_chart(
            chart_history_trend(history_summary_df),
            use_container_width=True,
        )

        st.dataframe(
            history_summary_df,
            use_container_width=True,
            hide_index=True,
        )
        return

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        start_date = st.selectbox(
            "Başlangıç tarihi",
            available_dates,
            index=0,
        )

    with c2:
        end_date = st.selectbox(
            "Bitiş tarihi",
            available_dates,
            index=len(available_dates) - 1,
        )

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

    if start_date == end_date:
        st.warning(
            "Başlangıç ve bitiş tarihi aynı seçildi. Değişim analizi için farklı iki tarih seç."
        )

    summary_compare = compare_summary(
        history_summary_df,
        start_date,
        end_date,
    )

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

    st.plotly_chart(
        chart_history_trend(history_summary_df),
        use_container_width=True,
    )

    st.plotly_chart(
        chart_history_metric(history_summary_df, selected_history_metric),
        use_container_width=True,
    )

    compare_df = compare_snapshot_dates(
        history_df,
        start_date,
        end_date,
    )

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

        st.plotly_chart(
            chart_record_delta(compare_df, delta_metric),
            use_container_width=True,
        )

        st.dataframe(
            compare_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_history_compare = compare_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Tarihsel Karşılaştırma CSV İndir",
            data=csv_history_compare,
            file_name=f"smartshelter_compare_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("#### Tüm Snapshot Özetleri")

    st.dataframe(
        history_summary_df,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Main App
# ---------------------------------------------------------
inject_css()
render_header()

st.sidebar.header("Veri Kaynağı")

mode = st.sidebar.radio(
    "Veri modu",
    [
        "Stabil Demo CSV",
        "Canlı CKAN API Dene",
    ],
)

raw_df = pd.DataFrame()
selected_source_name = "Stabil Demo CSV"
selected_resource_label = "Lokal CSV"
deep_scan = False

if mode == "Stabil Demo CSV":
    try:
        raw_df = load_local_data(LOCAL_FILE)
        st.success("Stabil demo CSV verisi kullanılıyor.")
    except Exception as e:
        st.error("Lokal CSV dosyası okunamadı.")
        st.exception(e)
        st.stop()

else:
    selected_source_name = st.sidebar.selectbox(
        "Canlı kaynak seç",
        list(CKAN_SOURCES.keys()),
    )

    source = CKAN_SOURCES[selected_source_name]

    deep_scan = st.sidebar.checkbox(
        "Derin geçmiş kaynak taraması",
        value=True,
        help=(
            "Daha fazla anahtar kelimeyle eski CKAN resource kayıtlarını da arar. "
            "Bu seçenek eski CSV/XLSX/JSON/ODS kaynaklarını bulma ihtimalini artırır."
        ),
    )

    try:
        resources = search_ckan_resources(
            source["base"],
            source["query"],
            rows=100 if deep_scan else 20,
            deep_queries=source.get("deep_queries", []) if deep_scan else None,
        )

        if not resources:
            st.warning(
                "Bu kaynakta uygun CSV/XLSX/JSON/ODS resource bulunamadı. Lokal veri kullanılıyor."
            )
            raw_df = load_local_data(LOCAL_FILE)
            selected_source_name = "Fallback Demo CSV"
            selected_resource_label = "Lokal CSV"
        else:
            resource_labels = [
                (
                    f"{r['package']} | {r['name']} | {r['format'].upper()} "
                    f"| Güncelleme: {r.get('resource_last_modified') or r.get('package_modified') or '-'} "
                    f"| Arama: {r.get('matched_query', '-')}"
                )
                for r in resources
            ]

            selected_resource_label = st.sidebar.selectbox(
                "Resource seç",
                resource_labels,
            )

            selected_resource = resources[
                resource_labels.index(selected_resource_label)
            ]

            df_resources = pd.DataFrame(resources)

            with st.sidebar.expander("Bulunan Resource Detayları"):
                st.dataframe(
                    df_resources[
                        [
                            "package",
                            "name",
                            "format",
                            "matched_query",
                            "package_modified",
                            "resource_last_modified",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            raw_df = load_resource(selected_resource)

            if raw_df.empty:
                st.warning(
                    "Seçilen canlı kaynak boş döndü. Lokal demo veri kullanılıyor."
                )
                raw_df = load_local_data(LOCAL_FILE)
                selected_source_name = "Fallback Demo CSV"
                selected_resource_label = "Lokal CSV"
            else:
                st.success(f"Canlı veri kaynağı yüklendi: {selected_source_name}")
                st.caption(selected_resource_label)

    except Exception as e:
        st.warning("Canlı veri çekilemedi. Lokal stabil veri kullanılıyor.")
        st.error(str(e))
        raw_df = load_local_data(LOCAL_FILE)
        selected_source_name = "Fallback Demo CSV"
        selected_resource_label = "Lokal CSV"


df = normalize_columns(raw_df)
df = calculate_risk(df)
df = create_action_recommendations(df)

# Günlük tarihsel snapshot kaydı.
# Aynı gün + aynı kaynak + aynı resource + aynı kayıt için mükerrer satır oluşmaz.
history_df = append_snapshot(
    df=df,
    source_name=selected_source_name,
    resource_label=selected_resource_label,
)

history_summary_df = build_history_summary(history_df)

st.sidebar.header("Filtreler")

districts = sorted(df["district"].dropna().astype(str).unique().tolist())
cities = sorted(df["city"].dropna().astype(str).unique().tolist())
risk_levels = ["Düşük", "Orta", "Kritik"]

selected_cities = st.sidebar.multiselect(
    "İl seç",
    cities,
    default=cities,
)

selected_districts = st.sidebar.multiselect(
    "İlçe seç",
    districts,
    default=districts,
)

selected_risks = st.sidebar.multiselect(
    "Risk seviyesi",
    risk_levels,
    default=risk_levels,
)

show_only_valid_coordinates = st.sidebar.checkbox(
    "Sadece koordinatı geçerli kayıtlar",
    value=False,
)

filtered_df = df[
    (df["city"].astype(str).isin(selected_cities))
    & (df["district"].astype(str).isin(selected_districts))
    & (df["risk_level"].astype(str).isin(selected_risks))
].copy()

if show_only_valid_coordinates:
    filtered_df = filtered_df[filtered_df["coordinate_valid"] == True].copy()  # noqa: E712

district_summary = build_district_summary(filtered_df)

with st.expander("ℹ️ Prototip Bilgisi", expanded=False):
    render_methodology_note()
    st.write(f"**Aktif veri kaynağı:** {selected_source_name}")
    st.write(f"**Resource:** {selected_resource_label}")
    st.write(f"**Derin CKAN taraması:** {'Açık' if deep_scan else 'Kapalı'}")
    st.write(
        "**Tarihsel snapshot dosyası:** `data/history/shelter_history.csv`"
    )

render_kpis(filtered_df)

st.divider()

left, right = st.columns([2.2, 1])

with left:
    st.subheader("📍 GIS Haritası")

    map_df = filtered_df[filtered_df["coordinate_valid"] == True].copy()  # noqa: E712

    if len(map_df) == 0:
        st.warning("Harita için geçerli koordinata sahip kayıt bulunamadı.")
    else:
        shelter_map = create_shelter_map(map_df)
        st_folium(
            shelter_map,
            width=1100,
            height=620,
            returned_objects=[],
        )

with right:
    render_record_detail(filtered_df)

st.divider()

st.subheader("📊 Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Risk Skoru",
        "Doluluk",
        "İlçe Özeti",
        "Geçmiş Analitik",
        "Veri Kalitesi",
        "Ham Veri",
        "Rapor",
        "Proje Vizyonu",
    ]
)

with tab1:
    st.plotly_chart(
        chart_risk_score(filtered_df),
        use_container_width=True,
    )

    st.markdown("#### Operasyonel Öncelik Listesi")

    priority_cols = [
        "name",
        "city",
        "district",
        "risk_level",
        "risk_score",
        "occupancy_rate",
        "animals_per_vet",
        "recommended_action",
    ]

    st.dataframe(
        filtered_df.sort_values("risk_score", ascending=False)[priority_cols],
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.plotly_chart(
        chart_occupancy_rate(filtered_df),
        use_container_width=True,
    )

with tab3:
    st.plotly_chart(
        chart_district_avg_risk(district_summary),
        use_container_width=True,
    )

    st.dataframe(
        district_summary,
        use_container_width=True,
        hide_index=True,
    )

with tab4:
    render_history_analytics(history_df, history_summary_df)

with tab5:
    render_data_quality_summary(filtered_df)

    quality_cols = [
        "name",
        "city",
        "district",
        "coordinate_valid",
        "is_estimated",
        "capacity_estimated",
        "occupancy_estimated",
        "vet_count_estimated",
        "data_quality_note",
    ]

    st.markdown("#### Veri Kalitesi Detayı")

    st.dataframe(
        filtered_df[quality_cols],
        use_container_width=True,
        hide_index=True,
    )

with tab6:
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
    )

with tab7:
    render_report_downloads(
        filtered_df,
        district_summary,
        history_df,
        history_summary_df,
    )

with tab8:
    st.markdown(
        """
        ### 🌍 SmartShelter GIS Vizyonu

        SmartShelter GIS; belediyeler, Tarım ve Orman Bakanlığı, STK'lar ve yerel yönetimler arasında
        ortak bir veri dili oluşturmayı hedefleyen açık veri tabanlı bir karar destek prototipidir.

        #### Temel amaçlar

        - Hayvan bakımevleri ve toplama merkezlerini harita üzerinde izlemek
        - Kapasite ve doluluk baskısını görünür hale getirmek
        - Veteriner iş yükünü ölçülebilir hale getirmek
        - Kısırlaştırma ve sahiplendirme performansını takip etmek
        - Kritik merkezleri önceliklendirmek
        - İl/ilçe bazlı operasyonel planlamaya destek olmak
        - Geçmiş veriler üzerinden değişim analizi yapmak
        - Eski/güncel açık veri kaynaklarını birlikte izlemek

        #### Geçmiş analitik yaklaşımı

        Sistem her veri çekiminde günlük bir snapshot oluşturur. Böylece şu sorular yanıtlanabilir:

        - Toplam kapasite önceki tarihe göre arttı mı?
        - Mevcut hayvan sayısı azaldı mı, arttı mı?
        - Ortalama risk hangi tarihte yükseldi?
        - Hangi merkezde risk skoru kötüleşti?
        - Hangi merkez yeni eklendi veya veri kaynağından çıktı?
        - Tahmini veri kullanılan kayıt sayısı zamanla azaldı mı?

        #### Önerilen sonraki aşamalar

        1. Ulusal veri standardı oluşturulması  
        2. Belediye sistemleriyle API entegrasyonu  
        3. PostGIS tabanlı merkezi coğrafi veri altyapısı  
        4. Mobil saha veri girişi  
        5. Gerçek zamanlı kapasite ve vaka takibi  
        6. Bakanlık düzeyinde izleme ve raporlama ekranı  
        7. Zaman serisi tabanlı erken uyarı sistemi  

        > Not: Bu uygulama resmi bir sistem değil, karar destek amaçlı çalışan prototip bir yazılımdır.
        """
    )

st.info(
    "Canlı API erişimi başarısız olursa sistem otomatik olarak lokal stabil CSV verisine döner. "
    "Her veri çekimi günlük snapshot olarak tarihsel arşive eklenir."
)
