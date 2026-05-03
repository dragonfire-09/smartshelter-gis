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

LOCAL_FILE = "data/kocaeli_shelters.csv"

CKAN_SOURCES = {
    "Kocaeli Açık Veri": {
        "base": "https://veri.kocaeli.bel.tr",
        "query": "hayvan toplama merkezi"
    },
    "Ordu Açık Veri": {
        "base": "https://acikveri.ordu.bel.tr",
        "query": "hayvan bakımevi"
    },
    "B40 İstanbul": {
        "base": "https://opendata.b40cities.org",
        "query": "hayvan bakımevi"
    }
}


@st.cache_data(ttl=3600)
def load_local_data():
    return pd.read_csv(LOCAL_FILE)


def safe_get_json(url, params=None, timeout=15):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600)
def search_ckan_resources(base_url, query):
    search_url = f"{base_url}/api/3/action/package_search"
    data = safe_get_json(search_url, params={"q": query, "rows": 10})
    results = data.get("result", {}).get("results", [])

    resources = []

    for package in results:
        for res in package.get("resources", []):
            fmt = str(res.get("format", "")).lower()
            url = res.get("url", "")
            name = res.get("name", package.get("title", "Veri"))

            if url and fmt in ["csv", "xlsx", "xls", "json", "ods"]:
                resources.append({
                    "package": package.get("title", ""),
                    "name": name,
                    "format": fmt,
                    "url": url,
                    "resource_id": res.get("id", "")
                })

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
                if "records" in data:
                    return pd.DataFrame(data["records"])
                if "result" in data and "records" in data["result"]:
                    return pd.DataFrame(data["result"]["records"])
                return pd.DataFrame([data])

    return pd.DataFrame()


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {}

    for c in df.columns:
        if c in ["adi", "ad", "tesis_adi", "merkez_adi", "barinak_adi", "bakimevi_adi"]:
            rename_map[c] = "name"
        elif c in ["ilce", "ilçe", "district"]:
            rename_map[c] = "district"
        elif c in ["il", "city"]:
            rename_map[c] = "city"
        elif c in ["enlem", "lat", "latitude", "y"]:
            rename_map[c] = "lat"
        elif c in ["boylam", "lon", "lng", "longitude", "x"]:
            rename_map[c] = "lon"
        elif "kapasite" in c:
            rename_map[c] = "capacity"
        elif "doluluk" in c or "mevcut" in c:
            rename_map[c] = "occupancy"
        elif "veteriner" in c:
            rename_map[c] = "vet_count"
        elif "kısır" in c or "kisir" in c:
            rename_map[c] = "sterilization_count"
        elif "sahip" in c:
            rename_map[c] = "adoption_count"

    df = df.rename(columns=rename_map)

    defaults = {
        "name": "Hayvan Bakımevi / Toplama Merkezi",
        "city": "Belirtilmemiş",
        "district": "Belirtilmemiş",
        "lat": None,
        "lon": None,
        "capacity": 100,
        "occupancy": 70,
        "vet_count": 1,
        "sterilization_count": 0,
        "adoption_count": 0
    }

    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    for col in ["lat", "lon", "capacity", "occupancy", "vet_count", "sterilization_count", "adoption_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["capacity"] = df["capacity"].fillna(100)
    df["occupancy"] = df["occupancy"].fillna(70)
    df["vet_count"] = df["vet_count"].fillna(1)
    df["sterilization_count"] = df["sterilization_count"].fillna(0)
    df["adoption_count"] = df["adoption_count"].fillna(0)

    return df


def calculate_risk(df):
    df = df.copy()
    df["occupancy_rate"] = (df["occupancy"] / df["capacity"].replace(0, 1) * 100).round(1)

    df["risk_score"] = (
        df["occupancy_rate"] * 0.60 +
        (100 / (df["vet_count"] + 1)) * 0.25 +
        ((df["capacity"] - df["adoption_count"]).clip(lower=0) / df["capacity"].replace(0, 1) * 15)
    ).round(1)

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, 40, 70, 999],
        labels=["Düşük", "Orta", "Kritik"]
    )

    return df


st.title("🐾 SmartShelter GIS")
st.caption("Gerçek veri entegrasyonu + fallback veri + GIS harita + dashboard + risk skoru")

st.sidebar.header("Veri Kaynağı")

mode = st.sidebar.radio(
    "Veri modu",
    [
        "Stabil Demo CSV",
        "Canlı CKAN API Dene"
    ]
)

df = pd.DataFrame()

if mode == "Stabil Demo CSV":
    df = load_local_data()
    st.success("Stabil GitHub CSV verisi kullanılıyor.")

else:
    selected_source = st.sidebar.selectbox(
        "Canlı kaynak seç",
        list(CKAN_SOURCES.keys())
    )

    source = CKAN_SOURCES[selected_source]

    try:
        resources = search_ckan_resources(source["base"], source["query"])

        if not resources:
            st.warning("Bu kaynakta uygun CSV/XLSX/JSON/ODS resource bulunamadı. Lokal veri kullanılıyor.")
            df = load_local_data()
        else:
            resource_labels = [
                f"{r['package']} | {r['name']} | {r['format'].upper()}"
                for r in resources
            ]

            selected_label = st.sidebar.selectbox("Resource seç", resource_labels)
            selected_resource = resources[resource_labels.index(selected_label)]

            df = load_resource(selected_resource)

            st.success(f"Canlı veri kaynağı yüklendi: {selected_source}")
            st.caption(selected_label)

    except Exception as e:
        st.warning("Canlı veri çekilemedi. Lokal stabil veri kullanılıyor.")
        st.error(str(e))
        df = load_local_data()


df = normalize_columns(df)
df = calculate_risk(df)

st.sidebar.header("Filtreler")

districts = sorted(df["district"].dropna().astype(str).unique().tolist())
risk_levels = ["Düşük", "Orta", "Kritik"]

selected_districts = st.sidebar.multiselect(
    "İlçe seç",
    districts,
    default=districts
)

selected_risks = st.sidebar.multiselect(
    "Risk seviyesi",
    risk_levels,
    default=risk_levels
)

filtered_df = df[
    (df["district"].astype(str).isin(selected_districts)) &
    (df["risk_level"].astype(str).isin(selected_risks))
]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Toplam Merkez / Kayıt", len(filtered_df))
col2.metric("Toplam Kapasite", int(filtered_df["capacity"].sum()))
col3.metric("Mevcut Hayvan", int(filtered_df["occupancy"].sum()))
col4.metric(
    "Ortalama Risk",
    f"{filtered_df['risk_score'].mean():.1f}" if len(filtered_df) else "0"
)

critical_count = len(filtered_df[filtered_df["risk_level"].astype(str) == "Kritik"])

if critical_count > 0:
    st.error(f"🔴 {critical_count} kayıt kritik risk seviyesinde.")
else:
    st.success("🟢 Kritik seviyede kayıt bulunmuyor.")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("📍 GIS Haritası")

    map_df = filtered_df.dropna(subset=["lat", "lon"])

    if len(map_df) == 0:
        st.warning("Bu veri kaynağında koordinat bulunamadı. Harita için lat/lon gerekli.")
    else:
        m = folium.Map(
            location=[map_df["lat"].mean(), map_df["lon"].mean()],
            zoom_start=10,
            tiles="OpenStreetMap"
        )

        for _, row in map_df.iterrows():
            color = "green"
            if row["risk_level"] == "Orta":
                color = "orange"
            elif row["risk_level"] == "Kritik":
                color = "red"

            popup = f"""
            <b>{row['name']}</b><br>
            İl: {row['city']}<br>
            İlçe: {row['district']}<br>
            Kapasite: {row['capacity']}<br>
            Mevcut: {row['occupancy']}<br>
            Doluluk: %{row['occupancy_rate']}<br>
            Veteriner: {row['vet_count']}<br>
            Kısırlaştırma: {row['sterilization_count']}<br>
            Sahiplendirme: {row['adoption_count']}<br>
            Risk: {row['risk_score']} - {row['risk_level']}
            """

            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(popup, max_width=350),
                tooltip=str(row["name"]),
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(m)

        st_folium(m, width=900, height=550)

with right:
    st.subheader("🏥 Kayıt Detayı")

    if len(filtered_df) > 0:
        selected_name = st.selectbox(
            "Kayıt seç",
            filtered_df["name"].astype(str).tolist()
        )

        item = filtered_df[filtered_df["name"].astype(str) == selected_name].iloc[0]

        st.write(f"**İl:** {item['city']}")
        st.write(f"**İlçe:** {item['district']}")
        st.write(f"**Kapasite:** {int(item['capacity'])}")
        st.write(f"**Mevcut Hayvan:** {int(item['occupancy'])}")
        st.write(f"**Doluluk Oranı:** %{item['occupancy_rate']}")
        st.write(f"**Veteriner Sayısı:** {int(item['vet_count'])}")
        st.write(f"**Kısırlaştırma:** {int(item['sterilization_count'])}")
        st.write(f"**Sahiplendirme:** {int(item['adoption_count'])}")
        st.write(f"**Risk Skoru:** {item['risk_score']}")

        if item["risk_level"] == "Kritik":
            st.error("🔴 Kritik risk seviyesi")
        elif item["risk_level"] == "Orta":
            st.warning("🟠 Orta risk seviyesi")
        else:
            st.success("🟢 Düşük risk seviyesi")

st.divider()

st.subheader("📊 Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Risk Skoru", "Doluluk", "Ham Veri", "Proje Vizyonu"]
)

with tab1:
    fig = px.bar(
        filtered_df,
        x="name",
        y="risk_score",
        color="risk_level",
        text="risk_score",
        title="Risk Skoru"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = px.bar(
        filtered_df,
        x="name",
        y="occupancy_rate",
        color="district",
        text="occupancy_rate",
        title="Doluluk Oranı (%)"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.dataframe(filtered_df, use_container_width=True)

with tab4:
    st.markdown("""
    ### 🌍 SmartShelter GIS Vizyonu

    Bu platform, belediyelerin açık veri kaynaklarını kullanarak hayvan barınakları ve sokak hayvanları yönetimini
    daha şeffaf, ölçülebilir ve veri odaklı hale getirmeyi amaçlar.

    Sistem; GIS haritalama, açık veri entegrasyonu, risk skorlama ve karar destek mekanizmalarını birleştirerek
    belediyeler, bakanlıklar, STK’lar ve vatandaşlar arasında daha etkili bir koordinasyon modeli sunar.
    """)

st.info("Canlı API erişimi başarısız olursa sistem otomatik olarak GitHub içindeki stabil CSV verisine döner.")
