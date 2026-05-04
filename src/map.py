import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _get_lat_col(df: pd.DataFrame) -> str:
    for cand in ["latitude", "lat", "enlem", "y"]:
        if cand in df.columns:
            return cand
    return None


def _get_lon_col(df: pd.DataFrame) -> str:
    for cand in ["longitude", "lon", "lng", "boylam", "x"]:
        if cand in df.columns:
            return cand
    return None


def _risk_color(level: str) -> str:
    level = str(level).strip()

    if level == "Kritik":
        return "red"
    if level == "Orta":
        return "orange"
    if level == "Düşük":
        return "green"
    return "gray"


def _risk_hex(level: str) -> str:
    level = str(level).strip()

    if level == "Kritik":
        return "#ef4444"
    if level == "Orta":
        return "#f59e0b"
    if level == "Düşük":
        return "#10b981"
    return "#94a3b8"


def _safe_int(val, default=0):
    try:
        if pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default


def _safe_float(val, default=0.0):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def create_shelter_map(df: pd.DataFrame) -> folium.Map:
    """
    Folium tabanlı GIS haritası oluşturur.
    'latitude'/'longitude' veya geriye dönük uyumluluk için 'lat'/'lon' kolonlarını destekler.
    """
    lat_col = _get_lat_col(df)
    lon_col = _get_lon_col(df)

    if lat_col is None or lon_col is None or df.empty:
        # Boş varsayılan harita - Türkiye merkezi
        return folium.Map(
            location=[39.0, 35.0],
            zoom_start=6,
            tiles="CartoDB positron",
        )

    work_df = df.copy()
    work_df[lat_col] = pd.to_numeric(work_df[lat_col], errors="coerce")
    work_df[lon_col] = pd.to_numeric(work_df[lon_col], errors="coerce")

    work_df = work_df.dropna(subset=[lat_col, lon_col])

    if work_df.empty:
        return folium.Map(
            location=[39.0, 35.0],
            zoom_start=6,
            tiles="CartoDB positron",
        )

    center_lat = work_df[lat_col].mean()
    center_lon = work_df[lon_col].mean()

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=None,
    )

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer("CartoDB positron", name="Açık Tema").add_to(fmap)
    folium.TileLayer("CartoDB dark_matter", name="Koyu Tema").add_to(fmap)

    shelter_layer = folium.FeatureGroup(name="Barınak / Bakımevi Noktaları", show=True)
    critical_layer = folium.FeatureGroup(name="Kritik Riskli Kayıtlar", show=True)
    heatmap_layer = folium.FeatureGroup(name="Risk Yoğunluk Haritası", show=True)

    cluster = MarkerCluster().add_to(shelter_layer)

    heat_points = []

    for _, row in work_df.iterrows():
        lat = _safe_float(row[lat_col])
        lon = _safe_float(row[lon_col])

        name = str(row.get("name", "Bilinmeyen"))
        city = str(row.get("city", "")) if pd.notna(row.get("city")) else ""
        district = str(row.get("district", "")) if pd.notna(row.get("district")) else ""

        risk_level = str(row.get("risk_level", "Veri yetersiz"))
        risk_score = row.get("risk_score", None)

        capacity = _safe_int(row.get("capacity"))
        occupancy = _safe_int(row.get("occupancy"))

        capacity_available = bool(row.get("capacity_available", False))
        occupancy_available = bool(row.get("occupancy_available", False))

        occupancy_rate = row.get("occupancy_rate", None)
        animals_per_vet = row.get("animals_per_vet", None)

        source_portal = str(row.get("source_portal", "")) if pd.notna(row.get("source_portal")) else ""

        # Popup içeriği
        popup_lines = [
            f"<b>{name}</b>",
            f"İl/İlçe: {city} / {district}".strip(" /"),
        ]

        if source_portal:
            popup_lines.append(f"Kaynak: {source_portal}")

        popup_lines.append(f"Risk: <b style='color:{_risk_hex(risk_level)}'>{risk_level}</b>")

        if pd.notna(risk_score):
            popup_lines.append(f"Risk Skoru: {risk_score}")

        if capacity_available:
            popup_lines.append(f"Kapasite: {capacity}")

        if occupancy_available:
            popup_lines.append(f"Mevcut: {occupancy}")

        if pd.notna(occupancy_rate):
            popup_lines.append(f"Doluluk: %{occupancy_rate}")

        if pd.notna(animals_per_vet):
            popup_lines.append(f"Veteriner başına: {animals_per_vet}")

        popup_html = "<br>".join(popup_lines)

        # Doluluk oranına göre marker boyutu
        if pd.notna(occupancy_rate):
            radius = max(5, min(18, 5 + float(occupancy_rate) / 8))
        else:
            radius = 6

        color = _risk_color(risk_level)
        hex_color = _risk_hex(risk_level)

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.75,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{name} - {risk_level}",
        ).add_to(cluster)

        if risk_level == "Kritik":
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="red", icon="exclamation-sign"),
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"KRİTİK: {name}",
            ).add_to(critical_layer)

        # Heatmap için risk skoru ağırlığı
        weight = _safe_float(risk_score, 0)
        if weight > 0:
            heat_points.append([lat, lon, weight / 100.0])

    if heat_points:
        HeatMap(
            heat_points,
            radius=22,
            blur=18,
            min_opacity=0.35,
            max_zoom=12,
        ).add_to(heatmap_layer)

    shelter_layer.add_to(fmap)
    critical_layer.add_to(fmap)
    heatmap_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    # Lejant
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px 14px;
        font-family: 'Segoe UI', sans-serif;
        font-size: 13px;
        box-shadow: 0 6px 20px rgba(15,23,42,0.12);
    ">
        <div style="font-weight:700; margin-bottom:6px;">Risk Seviyesi</div>
        <div style="margin:2px 0;"><span style="color:#10b981;">●</span> Düşük</div>
        <div style="margin:2px 0;"><span style="color:#f59e0b;">●</span> Orta</div>
        <div style="margin:2px 0;"><span style="color:#ef4444;">●</span> Kritik</div>
        <div style="margin:2px 0;"><span style="color:#94a3b8;">●</span> Veri yetersiz</div>
        <div style="margin-top:6px; color:#64748b; font-size:11px;">
            Marker boyutu doluluk oranına göre değişir.
        </div>
    </div>
    """

    fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap
