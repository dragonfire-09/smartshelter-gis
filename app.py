from io import BytesIO

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


st.set_page_config(
    page_title="SmartShelter GIS",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 17px;
            border-radius: 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }

        .small-note {
            color: #6b7280;
            font-size: 0.9rem;
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
        prototip bir karar destek ekranıdır.

        Türkiye geneli tarama modunda sistem yalnızca barınak/bakımevi envanteri olabilecek
        kaynakları ana risk dashboard'una almaya çalışır. İşlem sayısı, denetim istatistiği,
        evcil hayvan varlığı veya genel veteriner verileri ana kapasite/doluluk metriklerine
        karıştırılmaz.

        AI Analiz bölümü harici AI API kullanmadan, şeffaf ve kural tabanlı çalışır.
        """
    )


def render_kpis(df: pd.DataFrame):
    total_records = len(df)
    total_capacity = int(df["capacity"].sum()) if total_records else 0
    total_occupancy = int(df["occupancy"].sum()) if total_records else 0
    avg_risk = df["risk_score"].mean() if total_records else 0
    critical_count = len(df[df["risk_level"].astype(str) == "Kritik"])

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Analitik Kayıt", total_records)
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

    c1.metric("Analitik Kayıt", total)
    c2.metric("Geçersiz Koordinat", coord_missing)
    c3.metric("Tahmini Alan İçeren", estimated)
    c4.metric("Kapasite Tahmini", capacity_estimated)
    c5.metric("Veteriner Tahmini", vet_estimated)

    if estimated > 0:
        st.warning(
            "Bazı kayıtlarda eksik operasyonel alanlar demo/prototip amacıyla tahmini değerlerle tamamlanmıştır."
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

    if item.get("source_portal", ""):
        st.write(f"**Kaynak Portal:** {item.get('source_portal', '')}")

    if item.get("source_resource", ""):
        st.write(f"**Kaynak Resource:** {item.get('source_resource', '')}")

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

    if "risk_explanation" in item and str(item.get("risk_explanation", "")).strip():
        st.markdown("#### AI Risk Açıklaması")
        st.info(item["risk_explanation"])

    if "data_quality_score" in item:
        st.markdown("#### Veri Kalitesi Skoru")
        st.write(
            f"**{int(item['data_quality_score'])}/100** - {item.get('data_quality_level', '')}"
        )

    quality_note = str(item.get("data_quality_note", "")).strip()

    if quality_note:
        st.markdown("#### Veri Kalitesi Notu")
        st.warning(quality_note)


def render_history_analytics(history_df: pd.DataFrame, history_summary_df: pd.DataFrame):
    st.subheader("🕒 Geçmiş Analitik ve Tarihsel Karşılaştırma")

    available_dates = get_available_snapshot_dates(history_df)

    if len(available_dates) == 0:
        st.warning("Henüz tarihsel kayıt bulunmuyor.")
        return

    if len(available_dates) == 1:
        st.info(
            f"Şu anda yalnızca bir snapshot tarihi var: {available_dates[0]}. "
            "Karşılaştırma için farklı günlerde tekrar veri çekilmesi gerekir."
        )

        st.plotly_chart(chart_history_trend(history_summary_df), use_container_width=True)

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

    st.plotly_chart(chart_history_trend(history_summary_df), use_container_width=True)
    st.plotly_chart(
        chart_history_metric(history_summary_df, selected_history_metric),
        use_container_width=True,
    )

    compare_df = compare_snapshot_dates(history_df, start_date, end_date)

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

        st.plotly_chart(chart_record_delta(compare_df, delta_metric), use_container_width=True)

        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        csv_history_compare = compare_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Tarihsel Karşılaştırma CSV İndir",
            data=csv_history_compare,
            file_name=f"smartshelter_compare_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_ai_analysis(
    filtered_df: pd.DataFrame,
    history_summary_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
):
    st.subheader("🤖 AI Analiz ve Karar Destek")

    st.caption(
        "Bu bölüm harici AI API kullanmadan, kural tabanlı yönetici özeti, anomali tespiti ve senaryo analizi üretir."
    )

    ai_summary = generate_executive_summary(
        df=filtered_df,
        history_summary_df=history_summary_df,
        anomalies_df=anomalies_df,
    )

    st.markdown(ai_summary)

    st.divider()

    st.subheader("🚨 AI Uyarılar / Anomali Tespiti")

    if anomalies_df.empty:
        st.success("Belirgin anomali tespit edilmedi.")
    else:
        a1, a2, a3 = st.columns(3)

        critical_anomaly_count = len(
            anomalies_df[anomalies_df["severity"] == "Kritik"]
        )
        high_anomaly_count = len(
            anomalies_df[anomalies_df["severity"] == "Yüksek"]
        )
        total_anomaly_count = len(anomalies_df)

        a1.metric("Toplam Uyarı", total_anomaly_count)
        a2.metric("Yüksek Uyarı", high_anomaly_count)
        a3.metric("Kritik Uyarı", critical_anomaly_count)

        st.dataframe(anomalies_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("📊 Veri Kalitesi Skoru")

    if "data_quality_score" in filtered_df.columns and len(filtered_df) > 0:
        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "Ortalama Kalite",
            f"{filtered_df['data_quality_score'].mean():.1f}/100",
        )

        q2.metric(
            "Yüksek Kalite",
            len(filtered_df[filtered_df["data_quality_level"].astype(str) == "Yüksek"]),
        )

        q3.metric(
            "Orta Kalite",
            len(filtered_df[filtered_df["data_quality_level"].astype(str) == "Orta"]),
        )

        q4.metric(
            "Düşük Kalite",
            len(filtered_df[filtered_df["data_quality_level"].astype(str) == "Düşük"]),
        )

        quality_view_cols = [
            "name",
            "city",
            "district",
            "source_portal",
            "data_quality_score",
            "data_quality_level",
            "coordinate_valid",
            "capacity_estimated",
            "occupancy_estimated",
            "vet_count_estimated",
            "data_quality_note",
        ]

        quality_view_cols = [c for c in quality_view_cols if c in filtered_df.columns]

        st.dataframe(
            filtered_df.sort_values("data_quality_score")[quality_view_cols],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.subheader("🧪 Senaryo Simülasyonu")

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        extra_capacity = st.number_input(
            "+ Kapasite",
            min_value=0,
            max_value=5000,
            value=0,
            step=10,
        )

    with s2:
        extra_vets = st.number_input(
            "+ Veteriner",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )

    with s3:
        extra_adoptions = st.number_input(
            "+ Sahiplendirme",
            min_value=0,
            max_value=5000,
            value=0,
            step=10,
        )

    with s4:
        extra_sterilizations = st.number_input(
            "+ Kısırlaştırma",
            min_value=0,
            max_value=5000,
            value=0,
            step=10,
        )

    scenario_df = simulate_interventions(
        filtered_df,
        extra_capacity=extra_capacity,
        extra_vets=extra_vets,
        extra_adoptions=extra_adoptions,
        extra_sterilizations=extra_sterilizations,
    )

    if scenario_df.empty:
        st.warning("Senaryo için uygun kayıt bulunamadı.")
    else:
        avg_base_risk = scenario_df["base_risk_score"].mean()
        avg_scenario_risk = scenario_df["scenario_risk_score"].mean()
        avg_improvement = scenario_df["risk_score_improvement"].mean()

        c1, c2, c3 = st.columns(3)

        c1.metric("Mevcut Ortalama Risk", f"{avg_base_risk:.1f}")
        c2.metric("Senaryo Ortalama Risk", f"{avg_scenario_risk:.1f}")
        c3.metric("Ortalama İyileşme", f"{avg_improvement:.1f}")

        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

        scenario_csv = scenario_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Senaryo Sonucunu CSV İndir",
            data=scenario_csv,
            file_name="smartshelter_scenario_simulation.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_source_management(
    candidate_resources_df: pd.DataFrame,
    loaded_resources_info: pd.DataFrame,
    excluded_df: pd.DataFrame,
    df_all_loaded: pd.DataFrame,
):
    st.subheader("🧩 Kaynak Yönetimi ve Veri Ayrıştırma")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Aday Resource", len(candidate_resources_df))
    c2.metric("İçeri Alınan Resource", len(loaded_resources_info))
    c3.metric("Yüklenen Satır", len(df_all_loaded))
    c4.metric("Ana Analitikten Dışlanan", len(excluded_df))

    st.markdown("#### Resource Kategori Dağılımı")

    if not candidate_resources_df.empty and "resource_category" in candidate_resources_df.columns:
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

    else:
        st.info("Resource aday bilgisi yok.")

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

    st.markdown("#### Ana Analitikten Dışlanan Kayıtlar")

    if excluded_df.empty:
        st.success("Ana analitikten dışlanan kayıt yok.")
    else:
        display_cols = [
            "name",
            "city",
            "district",
            "source_portal",
            "source_resource",
            "resource_category",
            "analytics_exclusion_reason",
        ]

        display_cols = [c for c in display_cols if c in excluded_df.columns]

        st.warning(
            "Bu kayıtlar yüklendi ancak ana kapasite/doluluk/risk analitiğine uygun olmadığı için dışlandı."
        )

        st.dataframe(
            excluded_df[display_cols].head(500),
            use_container_width=True,
            hide_index=True,
        )


def render_report_downloads(
    df: pd.DataFrame,
    district_summary: pd.DataFrame,
    history_df: pd.DataFrame,
    history_summary_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
):
    st.subheader("📥 Rapor İndirme")

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

    if anomalies_df is not None and not anomalies_df.empty:
        anomalies_csv = anomalies_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="AI Anomali Raporu CSV İndir",
            data=anomalies_csv,
            file_name="smartshelter_ai_anomalies.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if excluded_df is not None and not excluded_df.empty:
        excluded_csv = excluded_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Dışlanan Kayıtlar CSV İndir",
            data=excluded_csv,
            file_name="smartshelter_excluded_records.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if history_df is not None and not history_df.empty:
        history_csv = history_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Tüm Tarihsel Snapshot Verisini CSV İndir",
            data=history_csv,
            file_name="smartshelter_history_snapshots.csv",
            mime="text/csv",
            use_container_width=True,
        )

        history_summary_csv = history_summary_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Tarihsel Özet CSV İndir",
            data=history_summary_csv,
            file_name="smartshelter_history_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
inject_css()
render_header()

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
        raw_df = load_local_data(LOCAL_FILE)
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

    try:
        resources = search_ckan_resources(
            source["base"],
            source["query"],
            rows=100 if deep_scan else 20,
            deep_queries=source.get("deep_queries", []) if deep_scan else None,
        )

        candidate_resources_df = pd.DataFrame(resources)

        if not resources:
            st.warning(
                "Bu kaynakta uygun resource bulunamadı. Lokal veri kullanılıyor."
            )
            raw_df = load_local_data(LOCAL_FILE)
            selected_source_name = "Fallback Demo CSV"
            selected_resource_label = "Lokal CSV"

        else:
            resource_labels = [
                (
                    f"[{r.get('resource_category', '-')}] "
                    f"{r['package']} | {r['name']} | {r['format'].upper()} "
                    f"| Skor: {r.get('relevance_score', 0)}"
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

            selected_resource["source_portal"] = selected_source_name

            with st.sidebar.expander("Bulunan Resource Detayları"):
                display_cols = [
                    "resource_category",
                    "relevance_score",
                    "package",
                    "name",
                    "format",
                    "matched_query",
                    "package_modified",
                    "resource_last_modified",
                ]

                existing_cols = [
                    c for c in display_cols if c in candidate_resources_df.columns
                ]

                st.dataframe(
                    candidate_resources_df[existing_cols],
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
        raw_df = load_local_data(LOCAL_FILE)
        selected_source_name = "Fallback Demo CSV"
        selected_resource_label = "Lokal CSV"


elif mode == "Türkiye Geneli CKAN Taraması":
    selected_source_name = "Türkiye Geneli CKAN Taraması"
    selected_resource_label = "Çoklu CKAN resource birleşimi"
    deep_scan = True

    st.sidebar.markdown("### Türkiye Geneli Tarama")

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
        "Sadece barınak/bakımevi envanteri kaynaklarını otomatik içeri al",
        value=True,
    )

    try:
        with st.spinner("Türkiye geneli CKAN portalları taranıyor..."):
            resources = search_turkiye_ckan_resources(
                rows_per_query=rows_per_query
            )

        candidate_resources_df = pd.DataFrame(resources)

        if not resources:
            st.warning(
                "Türkiye geneli taramada uygun resource bulunamadı. Lokal demo veri kullanılıyor."
            )
            raw_df = load_local_data(LOCAL_FILE)
            selected_source_name = "Fallback Demo CSV"
            selected_resource_label = "Lokal CSV"

        else:
            st.success(
                f"Türkiye geneli taramada {len(resources)} uygun resource adayı bulundu."
            )

            with st.expander("🇹🇷 Türkiye Geneli Bulunan Resource Adayları", expanded=False):
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

                existing_cols = [
                    c for c in display_cols if c in candidate_resources_df.columns
                ]

                st.dataframe(
                    candidate_resources_df[existing_cols],
                    use_container_width=True,
                    hide_index=True,
                )

            if auto_load:
                facility_resources = [
                    r for r in resources
                    if r.get("resource_category") == "shelter_facility"
                ]

                if not facility_resources:
                    st.warning(
                        "Barınak/bakımevi envanteri olarak sınıflanan resource bulunamadı. "
                        "Ana dashboard için lokal demo veri kullanılacak."
                    )
                    selected_resources = []
                else:
                    selected_resources = facility_resources[:max_resources_to_load]

            else:
                labels = [
                    (
                        f"[{r.get('resource_category', '-')}] "
                        f"{r.get('source_portal', '-')} | "
                        f"{r.get('package', '-')} | "
                        f"{r.get('name', '-')} | "
                        f"{r.get('format', '').upper()} | "
                        f"Skor: {r.get('relevance_score', 0)}"
                    )
                    for r in resources
                ]

                selected_labels = st.sidebar.multiselect(
                    "İçeri alınacak resource seç",
                    labels,
                    default=labels[: min(max_resources_to_load, len(labels))],
                )

                selected_resources = [
                    resources[labels.index(label)]
                    for label in selected_labels
                ]

            if not selected_resources:
                raw_df = load_local_data(LOCAL_FILE)
                selected_source_name = "Fallback Demo CSV"
                selected_resource_label = "Lokal CSV"

            else:
                with st.spinner("Seçilen resource dosyaları içeri alınıyor..."):
                    raw_df, loaded_resources, failed_resources = load_multiple_resources(
                        selected_resources,
                        max_resources=max_resources_to_load,
                        allowed_categories=["shelter_facility"],
                    )

                failed_resource_count = len(failed_resources)
                loaded_resources_info = pd.DataFrame(loaded_resources)

                if raw_df.empty:
                    st.warning(
                        "Seçilen barınak/bakımevi resource dosyaları okunamadı veya boş geldi. "
                        "Lokal demo veri kullanılıyor."
                    )
                    raw_df = load_local_data(LOCAL_FILE)
                    selected_source_name = "Fallback Demo CSV"
                    selected_resource_label = "Lokal CSV"
                else:
                    st.success(
                        f"{len(loaded_resources)} barınak/bakımevi resource başarıyla içeri alındı. "
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

                        existing_cols = [
                            c for c in display_cols if c in loaded_resources_info.columns
                        ]

                        st.dataframe(
                            loaded_resources_info[existing_cols],
                            use_container_width=True,
                            hide_index=True,
                        )

    except Exception as e:
        st.warning("Türkiye geneli tarama başarısız oldu. Lokal demo veri kullanılıyor.")
        st.error(str(e))
        raw_df = load_local_data(LOCAL_FILE)
        selected_source_name = "Fallback Demo CSV"
        selected_resource_label = "Lokal CSV"


# ---------------------------------------------------------
# Normalize + Eligibility
# ---------------------------------------------------------
df_all_loaded = normalize_columns(raw_df)

excluded_df = pd.DataFrame()

if "analytics_eligible" in df_all_loaded.columns:
    excluded_df = df_all_loaded[
        df_all_loaded["analytics_eligible"] == False  # noqa: E712
    ].copy()

    df = df_all_loaded[
        df_all_loaded["analytics_eligible"] == True  # noqa: E712
    ].copy()
else:
    df = df_all_loaded.copy()

if df.empty:
    st.warning(
        "Yüklenen kaynaklar ana barınak/bakımevi risk analitiği için uygun görünmüyor. "
        "Kapasite, mevcut hayvan ve merkez adı gibi temel alanlar eksik olabilir. "
        "Lokal demo veriye dönülüyor."
    )

    raw_df = load_local_data(LOCAL_FILE)
    selected_source_name = "Fallback Demo CSV"
    selected_resource_label = "Lokal CSV - analitik uygun kaynak bulunamadı"

    df_all_loaded = normalize_columns(raw_df)
    excluded_df = pd.DataFrame()
    df = df_all_loaded.copy()


# ---------------------------------------------------------
# Risk + AI + History
# ---------------------------------------------------------
df = calculate_risk(df)
df = create_action_recommendations(df)
df = calculate_data_quality_score(df)
df = generate_risk_explanations(df)

history_df = append_snapshot(
    df=df,
    source_name=selected_source_name,
    resource_label=selected_resource_label,
)

history_summary_df = build_history_summary(history_df)

anomalies_df = detect_anomalies(
    history_df=history_df,
    current_df=df,
)


# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------
st.sidebar.header("Filtreler")

cities = sorted(df["city"].dropna().astype(str).unique().tolist())
districts = sorted(df["district"].dropna().astype(str).unique().tolist())
risk_levels = ["Düşük", "Orta", "Kritik"]

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

filtered_df = df[
    (df["city"].astype(str).isin(selected_cities))
    & (df["district"].astype(str).isin(selected_districts))
    & (df["risk_level"].astype(str).isin(selected_risks))
].copy()

if show_only_valid_coordinates:
    filtered_df = filtered_df[
        filtered_df["coordinate_valid"] == True  # noqa: E712
    ].copy()

district_summary = build_district_summary(filtered_df)


# ---------------------------------------------------------
# Info Box + KPIs
# ---------------------------------------------------------
with st.expander("ℹ️ Prototip Bilgisi", expanded=False):
    render_methodology_note()

    st.write(f"**Aktif veri kaynağı:** {selected_source_name}")
    st.write(f"**Resource:** {selected_resource_label}")
    st.write(f"**Derin CKAN taraması:** {'Açık' if deep_scan else 'Kapalı'}")
    st.write("**Tarihsel snapshot dosyası:** `data/history/shelter_history.csv`")

    st.markdown("#### Veri Ayrıştırma Özeti")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Aday Resource",
        len(candidate_resources_df) if candidate_resources_df is not None else 0,
    )

    c2.metric(
        "İçeri Alınan Resource",
        len(loaded_resources_info) if loaded_resources_info is not None else 0,
    )

    c3.metric(
        "Yüklenen Satır",
        len(df_all_loaded) if df_all_loaded is not None else 0,
    )

    c4.metric(
        "Dışlanan Satır",
        len(excluded_df) if excluded_df is not None else 0,
    )

    if mode == "Türkiye Geneli CKAN Taraması":
        st.write(f"**Türkiye geneli kaynak sayısı:** {len(TURKIYE_CKAN_SOURCES)}")
        st.write(f"**Başarısız/boş resource sayısı:** {failed_resource_count}")

    if not excluded_df.empty:
        with st.expander("Ana analitikten dışlanan kayıt/resource örnekleri"):
            display_cols = [
                "name",
                "city",
                "district",
                "source_portal",
                "source_resource",
                "resource_category",
                "analytics_exclusion_reason",
            ]

            display_cols = [
                c for c in display_cols if c in excluded_df.columns
            ]

            st.dataframe(
                excluded_df[display_cols].head(300),
                use_container_width=True,
                hide_index=True,
            )


render_kpis(filtered_df)

st.divider()


# ---------------------------------------------------------
# Map + Record Detail
# ---------------------------------------------------------
left, right = st.columns([2.2, 1])

with left:
    st.subheader("📍 GIS Haritası")

    map_df = filtered_df[
        filtered_df["coordinate_valid"] == True  # noqa: E712
    ].copy()

    if len(map_df) == 0:
        st.warning(
            "Harita için geçerli koordinata sahip kayıt bulunamadı. "
            "Türkiye geneli kaynakların çoğunda koordinat alanı olmayabilir."
        )
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


# ---------------------------------------------------------
# Dashboard Tabs
# ---------------------------------------------------------
st.subheader("📊 Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
    [
        "Risk Skoru",
        "Doluluk",
        "İlçe Özeti",
        "Geçmiş Analitik",
        "AI Analiz",
        "Kaynak Yönetimi",
        "Veri Kalitesi",
        "Ham Veri",
        "Rapor",
        "Proje Vizyonu",
    ]
)


# ---------------------------------------------------------
# Tab 1 - Risk Score
# ---------------------------------------------------------
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
        "source_portal",
        "resource_category",
        "risk_level",
        "risk_score",
        "occupancy_rate",
        "animals_per_vet",
        "data_quality_score",
        "recommended_action",
        "risk_explanation",
    ]

    priority_cols = [
        c for c in priority_cols if c in filtered_df.columns
    ]

    if filtered_df.empty:
        st.warning("Filtreye uygun kayıt bulunamadı.")
    else:
        st.dataframe(
            filtered_df.sort_values(
                "risk_score",
                ascending=False,
            )[priority_cols],
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# Tab 2 - Occupancy
# ---------------------------------------------------------
with tab2:
    st.plotly_chart(
        chart_occupancy_rate(filtered_df),
        use_container_width=True,
    )

    st.markdown("#### Doluluk Detay Tablosu")

    occupancy_cols = [
        "name",
        "city",
        "district",
        "capacity",
        "occupancy",
        "occupancy_rate",
        "risk_level",
        "risk_score",
        "source_portal",
    ]

    occupancy_cols = [
        c for c in occupancy_cols if c in filtered_df.columns
    ]

    if filtered_df.empty:
        st.warning("Filtreye uygun kayıt bulunamadı.")
    else:
        st.dataframe(
            filtered_df.sort_values(
                "occupancy_rate",
                ascending=False,
            )[occupancy_cols],
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# Tab 3 - District Summary
# ---------------------------------------------------------
with tab3:
    st.plotly_chart(
        chart_district_avg_risk(district_summary),
        use_container_width=True,
    )

    st.markdown("#### İlçe Bazlı Özet")

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
        "data_quality_score",
        "data_quality_level",
        "coordinate_valid",
        "analytics_eligible",
        "capacity_estimated",
        "occupancy_estimated",
        "vet_count_estimated",
        "sterilization_count_estimated",
        "adoption_count_estimated",
        "data_quality_note",
    ]

    quality_cols = [
        c for c in quality_cols if c in filtered_df.columns
    ]

    if filtered_df.empty:
        st.warning("Filtreye uygun kayıt bulunamadı.")
    else:
        st.dataframe(
            filtered_df.sort_values(
                "data_quality_score",
                ascending=True,
            )[quality_cols],
            use_container_width=True,
            hide_index=True,
        )

    if not excluded_df.empty:
        st.markdown("#### Ana Analitikten Dışlanan Kayıtlar")

        excluded_cols = [
            "name",
            "city",
            "district",
            "source_portal",
            "source_resource",
            "resource_category",
            "analytics_exclusion_reason",
            "data_quality_note",
        ]

        excluded_cols = [
            c for c in excluded_cols if c in excluded_df.columns
        ]

        st.warning(
            "Aşağıdaki kayıtlar yüklendi ancak ana kapasite/doluluk/risk "
            "analitiğine uygun görülmediği için KPI ve risk hesaplarına dahil edilmedi."
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
    st.markdown("#### Ana Analitik Veri Seti")

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
# Tab 10 - Project Vision
# ---------------------------------------------------------
with tab10:
    st.markdown(
        """
        ### 🌍 SmartShelter GIS Vizyonu

        SmartShelter GIS; belediyeler, Tarım ve Orman Bakanlığı, STK'lar ve yerel yönetimler arasında
        ortak bir veri dili oluşturmayı hedefleyen açık veri tabanlı bir GIS karar destek prototipidir.

        #### Temel Amaçlar

        - Hayvan bakımevleri ve toplama merkezlerini harita üzerinde izlemek
        - Kapasite ve doluluk baskısını görünür hale getirmek
        - Veteriner iş yükünü ölçülebilir hale getirmek
        - Kısırlaştırma ve sahiplendirme performansını takip etmek
        - Kritik merkezleri önceliklendirmek
        - İl/ilçe bazlı operasyonel planlamaya destek olmak
        - Geçmiş veriler üzerinden değişim analizi yapmak
        - Türkiye geneli açık veri kaynaklarını sınıflandırarak analiz etmek
        - AI benzeri karar destek analizleri üretmek

        #### Kaynak Ayrıştırma Yaklaşımı

        Türkiye geneli açık veri portallarında "hayvan" veya "veteriner" geçen çok sayıda farklı veri seti bulunur.
        Bu veri setlerinin tamamı barınak kapasite/doluluk analizi için uygun değildir.

        Bu nedenle sistem kaynakları şu kategorilere ayırır:

        - **shelter_facility:** Barınak, bakımevi, geçici hayvan bakım merkezi gibi ana envanter kaynakları
        - **operation_stats:** İşlem sayısı, yıllık istatistik, denetim, evcil hayvan varlığı gibi operasyonel kaynaklar
        - **general_animal:** Hayvan/veteriner konulu ama envanter olduğu net olmayan kaynaklar
        - **irrelevant:** Ana konu ile ilgisiz kaynaklar

        Ana risk dashboard'una yalnızca analitik olarak uygun görülen kayıtlar dahil edilir.
        Kapasite ve mevcut hayvan alanları olmayan büyük istatistik dosyaları KPI hesaplarını şişirmemek için dışlanır.

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

        Sistem her veri çekiminde günlük bir snapshot oluşturur. Böylece şu sorular yanıtlanabilir:

        - Toplam kapasite önceki tarihe göre arttı mı?
        - Mevcut hayvan sayısı azaldı mı, arttı mı?
        - Ortalama risk hangi tarihte yükseldi?
        - Hangi merkezde risk skoru kötüleşti?
        - Hangi merkez yeni eklendi veya veri kaynağından çıktı?
        - Tahmini veri kullanılan kayıt sayısı zamanla azaldı mı?

        #### Önerilen Sonraki Aşamalar

        1. Ulusal veri standardı oluşturulması  
        2. Belediye sistemleriyle API entegrasyonu  
        3. PostGIS tabanlı merkezi coğrafi veri altyapısı  
        4. Mobil saha veri girişi  
        5. Gerçek zamanlı kapasite ve vaka takibi  
        6. Bakanlık düzeyinde izleme ve raporlama ekranı  
        7. Zaman serisi tabanlı erken uyarı sistemi  
        8. Operasyonel istatistik kaynakları için ayrı analiz sekmesi  
        9. CKAN dışı belediye web sayfaları için kontrollü scraping modülü  
        10. LLM destekli doğal dil sorgulama ve otomatik raporlama  

        > Not: Bu uygulama resmi bir sistem değil, karar destek amaçlı çalışan prototip bir yazılımdır.
        """
    )


st.info(
    "Canlı API erişimi başarısız olursa sistem otomatik olarak lokal stabil CSV verisine döner. "
    "Türkiye geneli taramada işlem/istatistik kaynakları ana KPI hesaplarına karıştırılmaz. "
    "Her veri çekimi günlük snapshot olarak tarihsel arşive eklenir."
)
