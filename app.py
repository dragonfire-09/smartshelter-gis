import streamlit as st
import pandas as pd
import requests
import folium
import plotly.express as px
from streamlit_folium import st_folium

st.set_page_config(
    page_title="SmartShelter GIS",
    page_icon="🐾",
    layout="wide"
)

st.title("🐾 SmartShelter GIS")
st.caption("Kocaeli açık veri entegrasyonu ile hayvan toplama merkezleri için GIS + dashboard + risk skoru prototipi")

RESOURCE_URL = "https://ulasav.csb.gov.tr/api/3/action/package_show?id=41-hayvan-toplama-merkezi"
RESOURCE_ID = "42045805-70d1-49f4-87cf-63fb50941952"


@st.cache_data(ttl=3600)
def load_kocaeli_data():
    try:
        response = requests.get(RESOURCE_URL, timeout=20)
        response.raise_for_status()
        package = response.json()

        resources = package["result"]["resources"]
        resource = [r for r in resources if r["id"] == RESOURCE_ID][0]
        file_url = resource["url"]

        df = pd.read_json(file_url)
        return df

    except Exception as e:
        st.warning("Canlı veri çekilemedi. Demo veri kullanılıyor.")
        st.error(str(e))

        return pd.DataFrame({
            "name": [
                "Kocaeli Sokak Hayvanları Toplama Merkezi",
                "Gebze Hayvan Toplama Merkezi",
                "İzmit Geçici Hayvan Bakımevi"
            ],
            "city": ["Kocaeli", "Kocaeli", "Kocaeli"],
            "district": ["İzmit", "Gebze", "İzmit"],
            "lat": [40.7654, 40.8028, 40.7666],
            "lon": [29.9408, 29.4307, 29.9169],
            "capacity": [500, 300, 250],
            "occupancy": [430, 285, 180],
            "vet_count": [4, 2, 3],
            "sterilization_count": [1200, 850, 640],
            "adoption_count": [320, 180, 210]
        })


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    possible_lat = ["lat", "latitude", "enlem", "y"]
    possible_lon = ["lon", "lng", "longitude", "boylam", "x"]
    possible_name = ["name", "adi", "ad", "tesis_adi", "merkez_adi"]

    for col in possible_lat:
        if col in df.columns:
            df["lat"] = pd.to_numeric(df[col], errors="coerce")
            break

    for col in possible_lon:
        if col in df.columns:
            df["lon"] = pd.to_numeric(df[col], errors="coerce")
            break

    for col in possible_name:
        if col in df.columns:
            df["name"] = df[col].astype(str)
            break

    if "name" not in df.columns:
        df["name"] = "Hayvan Toplama Merkezi"

    if "city" not in df.columns:
        df["city"] = "Kocaeli"

    if "district" not in df.columns:
        df["district"] = "Belirtilmemiş"

    if "capacity" not in df.columns:
        df["capacity"] = 100

    if "occupancy" not in df.columns:
        df["occupancy"] = 70

    if "vet_count" not in df.columns:
        df["vet_count"] = 1

    if "sterilization_count" not in df.columns:
        df["sterilization_count"] = 0

    if "adoption_count" not in df.columns:
        df["adoption_count"] = 0

    numeric_cols = ["capacity", "occupancy", "vet_count", "sterilization_count", "adoption_count"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def calculate_risk(df):
    df = df.copy()

    df["occupancy_rate"] = (df["occupancy"] / df["capacity"].replace(0, 1) * 100).round(1)

    df["risk_score"] = (
        df["occupancy_rate"] * 0.55 +
        (100 / (df["vet_count"] + 1)) * 0.25 +
        (df["occupancy"] / df["capacity"].replace(0, 1) * 20) * 0.20
    ).round(1)

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, 40, 70, 999],
        labels=["Düşük", "Orta", "Kritik"]
    )

    return df


raw_df = load_kocaeli_data()
df = normalize_columns(raw_df)
df = calculate_risk(df)

st.sidebar.header("Filtreler")

districts = sorted(df["district"].dropna().unique().tolist())
selected_districts = st.sidebar.multiselect(
    "İlçe seç",
    districts,
    default=districts
)

risk_levels = ["Düşük", "Orta", "Kritik"]
selected_risks = st.sidebar.multiselect(
    "Risk seviyesi",
    risk_levels,
    default=risk_levels
)

filtered_df = df[
    (df["district"].isin(selected_districts)) &
    (df["risk_level"].astype(str).isin(selected_risks))
]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Toplam Merkez", len(filtered_df))
col2.metric("Toplam Kapasite", int(filtered_df["capacity"].sum()))
col3.metric("Mevcut Hayvan", int(filtered_df["occupancy"].sum()))
col4.metric("Ortalama Risk", f"{filtered_df['risk_score'].mean():.1f}" if len(filtered_df) else "0")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("📍 GIS Haritası")

    map_df = filtered_df.dropna(subset=["lat", "lon"])

    if len(map_df) == 0:
        st.warning("Bu veri setinde koordinat bulunamadı. Harita için lat/lon bilgisi gerekiyor.")
    else:
        center_lat = map_df["lat"].mean()
        center_lon = map_df["lon"].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles="OpenStreetMap"
        )

        for _, row in map_df.iterrows():
            if row["risk_level"] == "Kritik":
                color = "red"
            elif row["risk_level"] == "Orta":
                color = "orange"
            else:
                color = "green"

            popup = f"""
            <b>{row['name']}</b><br>
            İlçe: {row['district']}<br>
            Kapasite: {row['capacity']}<br>
            Mevcut Hayvan: {row['occupancy']}<br>
            Doluluk: %{row['occupancy_rate']}<br>
            Veteriner Sayısı: {row['vet_count']}<br>
            Kısırlaştırma: {row['sterilization_count']}<br>
            Sahiplendirme: {row['adoption_count']}<br>
            Risk Skoru: {row['risk_score']}<br>
            Risk Seviyesi: {row['risk_level']}
            """

            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(popup, max_width=350),
                tooltip=row["name"],
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(m)

        st_folium(m, width=900, height=550)

with right:
    st.subheader("🏥 Merkez Detayı")

    if len(filtered_df) > 0:
        selected_name = st.selectbox(
            "Merkez seç",
            filtered_df["name"].tolist()
        )

        shelter = filtered_df[filtered_df["name"] == selected_name].iloc[0]

        st.write(f"**İlçe:** {shelter['district']}")
        st.write(f"**Kapasite:** {int(shelter['capacity'])}")
        st.write(f"**Mevcut Hayvan:** {int(shelter['occupancy'])}")
        st.write(f"**Doluluk Oranı:** %{shelter['occupancy_rate']}")
        st.write(f"**Veteriner Sayısı:** {int(shelter['vet_count'])}")
        st.write(f"**Kısırlaştırma:** {int(shelter['sterilization_count'])}")
        st.write(f"**Sahiplendirme:** {int(shelter['adoption_count'])}")
        st.write(f"**Risk Skoru:** {shelter['risk_score']}")

        if shelter["risk_level"] == "Kritik":
            st.error("🔴 Kritik risk seviyesi")
        elif shelter["risk_level"] == "Orta":
            st.warning("🟠 Orta risk seviyesi")
        else:
            st.success("🟢 Düşük risk seviyesi")

st.divider()

st.subheader("📊 Dashboard")

tab1, tab2, tab3 = st.tabs(["Risk Skoru", "Doluluk", "Veri Tablosu"])

with tab1:
    fig_risk = px.bar(
        filtered_df,
        x="name",
        y="risk_score",
        color="risk_level",
        title="Merkez Bazlı Risk Skoru",
        text="risk_score"
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with tab2:
    fig_occ = px.bar(
        filtered_df,
        x="name",
        y="occupancy_rate",
        color="district",
        title="Doluluk Oranı (%)",
        text="occupancy_rate"
    )
    st.plotly_chart(fig_occ, use_container_width=True)

with tab3:
    st.dataframe(filtered_df, use_container_width=True)

st.info(
    "Bu prototip; açık veri, GIS haritalama, temel risk skoru ve karar destek mantığını göstermek için hazırlanmıştır."
)
