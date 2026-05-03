import folium
from folium.plugins import Fullscreen, HeatMap, MarkerCluster, MeasureControl, MiniMap


def risk_color(risk_level):
    risk_level = str(risk_level)

    if risk_level == "Kritik":
        return "red"

    if risk_level == "Orta":
        return "orange"

    return "green"


def create_popup_html(row):
    quality_note = row.get("data_quality_note", "")

    if not quality_note:
        quality_note = "Veri kalite notu yok."

    html = f"""
    <div style="font-family: Arial; width: 300px;">
        <h4 style="margin-bottom: 8px;">{row["name"]}</h4>

        <b>İl:</b> {row["city"]}<br>
        <b>İlçe:</b> {row["district"]}<br>
        <b>Kapasite:</b> {int(row["capacity"])}<br>
        <b>Mevcut:</b> {int(row["occupancy"])}<br>
        <b>Doluluk:</b> %{row["occupancy_rate"]}<br>
        <b>Veteriner:</b> {int(row["vet_count"])}<br>
        <b>Veteriner Başına Hayvan:</b> {row["animals_per_vet"]}<br>
        <b>Kısırlaştırma:</b> {int(row["sterilization_count"])}<br>
        <b>Sahiplendirme:</b> {int(row["adoption_count"])}<br>

        <hr style="margin: 8px 0;">

        <b>Risk:</b> {row["risk_score"]} - {row["risk_level"]}<br>
        <b>Öneri:</b> {row.get("recommended_action", "-")}<br>

        <hr style="margin: 8px 0;">

        <small><b>Veri Kalitesi:</b> {quality_note}</small>
    </div>
    """

    return html


def add_legend(m):
    legend_html = """
    <div style="
        position: fixed;
        bottom: 35px;
        left: 35px;
        z-index: 9999;
        background-color: white;
        padding: 12px 14px;
        border: 1px solid #d1d5db;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        font-size: 13px;
    ">
        <b>Risk Seviyesi</b><br>
        <span style="color: green;">●</span> Düşük<br>
        <span style="color: orange;">●</span> Orta<br>
        <span style="color: red;">●</span> Kritik<br>
        <hr style="margin: 6px 0;">
        <small>Marker boyutu doluluk oranına göre değişir.</small>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))


def create_shelter_map(df):
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        "CartoDB positron",
        name="Açık Tema",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        "CartoDB dark_matter",
        name="Koyu Tema",
        control=True,
    ).add_to(m)

    Fullscreen(
        position="topleft",
        title="Tam ekran",
        title_cancel="Tam ekrandan çık",
    ).add_to(m)

    MiniMap(
        toggle_display=True,
        minimized=True,
        position="bottomright",
    ).add_to(m)

    MeasureControl(
        position="bottomleft",
        primary_length_unit="kilometers",
        secondary_length_unit="meters",
    ).add_to(m)

    marker_cluster = MarkerCluster(
        name="Barınak / Bakımevi Noktaları"
    ).add_to(m)

    critical_group = folium.FeatureGroup(
        name="Kritik Riskli Kayıtlar",
        show=True,
    ).add_to(m)

    for _, row in df.iterrows():
        color = risk_color(row["risk_level"])

        radius = 7 + min(float(row["occupancy_rate"]) / 10, 14)

        popup_html = create_popup_html(row)

        marker = folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=2,
            popup=folium.Popup(popup_html, max_width=360),
            tooltip=f"{row['name']} | Risk: {row['risk_level']} | Skor: {row['risk_score']}",
        )

        marker.add_to(marker_cluster)

        if str(row["risk_level"]) == "Kritik":
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=radius + 3,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.35,
                weight=3,
                popup=folium.Popup(popup_html, max_width=360),
                tooltip=f"KRİTİK: {row['name']}",
            ).add_to(critical_group)

    heat_data = (
        df[["lat", "lon", "risk_score"]]
        .dropna()
        .values
        .tolist()
    )

    if len(heat_data) > 0:
        HeatMap(
            heat_data,
            name="Risk Yoğunluk Haritası",
            min_opacity=0.25,
            radius=28,
            blur=20,
            max_zoom=13,
        ).add_to(m)

    add_legend(m)

    folium.LayerControl(
        collapsed=False,
        position="topright",
    ).add_to(m)

    return m
