"""
SmartShelter GIS - Harita Modülü
=================================
Folium tabanlı interaktif harita: çoklu tema + uydu görünümü +
risk seviyesine göre renk + katman kontrolü.
"""
import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd

# Türkiye merkez koordinatları
TURKEY_CENTER = [39.0, 35.0]
DEFAULT_ZOOM = 6

# Risk seviyesi → renk eşleme
RISK_COLORS = {
    "Düşük": "#22c55e",        # yeşil
    "Orta": "#f59e0b",          # turuncu
    "Yüksek": "#ef4444",        # kırmızı
    "Kritik": "#991b1b",        # koyu kırmızı
    "Veri yetersiz": "#94a3b8",  # gri
}


def _safe_float(val, default=None):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_str(val, default="—"):
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return default
    return s


def _compute_center(df: pd.DataFrame):
    """Veri ortasını bul, yoksa Türkiye ortası."""
    if df is None or df.empty:
        return TURKEY_CENTER, DEFAULT_ZOOM

    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "lon" if "lon" in df.columns else "longitude"

    if lat_col not in df.columns or lon_col not in df.columns:
        return TURKEY_CENTER, DEFAULT_ZOOM

    lat_s = pd.to_numeric(df[lat_col], errors="coerce")
    lon_s = pd.to_numeric(df[lon_col], errors="coerce")

    valid = lat_s.notna() & lon_s.notna()
    if not valid.any():
        return TURKEY_CENTER, DEFAULT_ZOOM

    center = [float(lat_s[valid].mean()), float(lon_s[valid].mean())]

    # Zoom: dağılıma göre otomatik
    lat_range = float(lat_s[valid].max() - lat_s[valid].min())
    lon_range = float(lon_s[valid].max() - lon_s[valid].min())
    spread = max(lat_range, lon_range)

    if spread > 5:
        zoom = 6
    elif spread > 2:
        zoom = 8
    elif spread > 0.5:
        zoom = 10
    elif spread > 0.1:
        zoom = 12
    else:
        zoom = 13

    return center, zoom


def _add_tile_layers(m):
    """Birden fazla harita teması ekle."""
    # Standart OpenStreetMap (default açık tema)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="🗺️ OpenStreetMap",
        attr="© OpenStreetMap contributors",
    ).add_to(m)

    # CartoDB Positron — açık tema
    folium.TileLayer(
        tiles="CartoDB positron",
        name="☀️ Açık Tema",
        attr="© OpenStreetMap © CARTO",
    ).add_to(m)

    # CartoDB Dark Matter — koyu tema
    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="🌙 Koyu Tema",
        attr="© OpenStreetMap © CARTO",
    ).add_to(m)

    # Esri World Imagery — uydu görünümü
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="🛰️ Uydu Görüntüsü",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
    ).add_to(m)

    # Esri WorldStreetMap — sokak haritası
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        name="🏙️ Sokak Haritası",
        attr="Tiles © Esri",
    ).add_to(m)

    # Stamen Terrain — arazi haritası
    folium.TileLayer(
        tiles="https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}{r}.png",
        name="⛰️ Arazi Haritası",
        attr="© Stadia Maps © Stamen Design © OpenStreetMap",
    ).add_to(m)


def _build_popup(row):
    """Marker popup HTML'i oluştur."""
    name = _safe_str(row.get("name"))
    city = _safe_str(row.get("city"))
    district = _safe_str(row.get("district"))
    capacity = row.get("capacity")
    occupancy = row.get("occupancy")
    risk_score = row.get("risk_score")
    risk_level = _safe_str(row.get("risk_level"))
    portal = _safe_str(row.get("source_portal"))

    cap_txt = f"{int(capacity):,}" if pd.notna(capacity) else "—"
    occ_txt = f"{int(occupancy):,}" if pd.notna(occupancy) else "—"
    risk_txt = f"{float(risk_score):.1f}" if pd.notna(risk_score) else "—"

    return f"""
    <div style='font-family: -apple-system, sans-serif; min-width: 240px;'>
        <h4 style='margin: 0 0 8px 0; color: #1e3a8a;'>🏥 {name}</h4>
        <p style='margin: 4px 0;'><b>📍 Konum:</b> {city} / {district}</p>
        <p style='margin: 4px 0;'><b>🏠 Kapasite:</b> {cap_txt}</p>
        <p style='margin: 4px 0;'><b>🐾 Mevcut Hayvan:</b> {occ_txt}</p>
        <p style='margin: 4px 0;'><b>⚠️ Risk Skoru:</b> {risk_txt} <span style='color:{RISK_COLORS.get(risk_level, "#666")}'>({risk_level})</span></p>
        <p style='margin: 4px 0; font-size: 11px; color: #666;'><b>📊 Kaynak:</b> {portal}</p>
    </div>
    """


def _add_marker_layers(m, df: pd.DataFrame):
    """Marker katmanlarını ekle: tüm noktalar, kritik, ve cluster."""
    if df is None or df.empty:
        return

    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "lon" if "lon" in df.columns else "longitude"

    if lat_col not in df.columns or lon_col not in df.columns:
        return

    # Layer'lar
    all_markers = folium.FeatureGroup(name="📍 Tüm Barınaklar", show=True)
    cluster_group = MarkerCluster(name="🔵 Cluster (Yoğunluk)", show=False)
    critical_markers = folium.FeatureGroup(name="🚨 Kritik Riskli", show=True)
    heatmap_data = []

    for _, row in df.iterrows():
        lat = _safe_float(row.get(lat_col))
        lon = _safe_float(row.get(lon_col))

        if lat is None or lon is None:
            continue
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            continue

        risk_level = _safe_str(row.get("risk_level"), "Veri yetersiz")
        color = RISK_COLORS.get(risk_level, "#94a3b8")

        # Marker boyutu kapasite/doluluk oranına göre
        capacity = row.get("capacity")
        occupancy = row.get("occupancy")
        radius = 6
        if pd.notna(capacity) and pd.notna(occupancy) and capacity > 0:
            try:
                ratio = min(float(occupancy) / float(capacity), 2.0)
                radius = 5 + int(ratio * 8)  # 5-21 arası
            except Exception:
                pass

        popup_html = _build_popup(row)
        tooltip = _safe_str(row.get("name"))

        # Tüm marker katmanı
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=tooltip,
        ).add_to(all_markers)

        # Cluster için aynı marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=tooltip,
        ).add_to(cluster_group)

        # Kritik riskli ayrı katman
        if risk_level == "Kritik":
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="red", icon="warning-sign"),
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"🚨 KRİTİK: {tooltip}",
            ).add_to(critical_markers)

        # Heatmap verisi
        weight = 1.0
        if pd.notna(row.get("risk_score")):
            try:
                weight = max(float(row["risk_score"]) / 100.0, 0.3)
            except Exception:
                pass
        heatmap_data.append([lat, lon, weight])

    # Layer'ları haritaya ekle
    all_markers.add_to(m)
    critical_markers.add_to(m)
    cluster_group.add_to(m)

    # Heatmap katmanı
    if heatmap_data:
        heatmap_layer = folium.FeatureGroup(name="🔥 Risk Yoğunluk Haritası", show=False)
        HeatMap(
            heatmap_data,
            min_opacity=0.3,
            radius=20,
            blur=15,
            max_zoom=12,
        ).add_to(heatmap_layer)
        heatmap_layer.add_to(m)


def _add_legend(m):
    """Sol alt köşeye renk legend'ı ekle."""
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 1000;
        background: white;
        padding: 12px 16px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-family: -apple-system, sans-serif;
        font-size: 13px;
    ">
        <b style="font-size: 14px;">Risk Seviyesi</b><br>
        <span style="color:#22c55e;">●</span> Düşük<br>
        <span style="color:#f59e0b;">●</span> Orta<br>
        <span style="color:#ef4444;">●</span> Yüksek<br>
        <span style="color:#991b1b;">●</span> Kritik<br>
        <span style="color:#94a3b8;">●</span> Veri yetersiz<br>
        <span style="color:#666; font-size: 11px;">
            Marker boyutu doluluk oranına göre değişir.
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def create_shelter_map(df: pd.DataFrame, default_tile="OpenStreetMap"):
    """SmartShelter GIS interaktif harita oluştur.
    
    Özellikler:
    - 6 farklı harita teması (OSM, Açık, Koyu, Uydu, Sokak, Arazi)
    - 4 katman (Tüm Barınaklar, Cluster, Kritik Riskli, Heatmap)
    - Risk seviyesine göre renkli marker'lar
    - Doluluk oranına göre marker boyutu
    - Detaylı popup'lar
    - Sağ üstte katman kontrolü
    - Sol altta renk legend'ı
    """
    center, zoom = _compute_center(df)

    # Boş başlat — ilk tile layer'ı manuel ekleyeceğiz
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,  # Önce boş, _add_tile_layers ekleyecek
        control_scale=True,
    )

    # Harita temalarını ekle
    _add_tile_layers(m)

    # Marker katmanlarını ekle
    _add_marker_layers(m, df)

    # Katman kontrolü (sağ üst köşe)
    folium.LayerControl(
        position="topright",
        collapsed=False,  # Açık başlasın ki kullanıcı görsün
    ).add_to(m)

    # Legend
    _add_legend(m)

    # Tam ekran butonu (opsiyonel ama güzel)
    try:
        from folium.plugins import Fullscreen
        Fullscreen(
            position="topleft",
            title="Tam ekran",
            title_cancel="Çıkış",
            force_separate_button=True,
        ).add_to(m)
    except Exception:
        pass

    return m
