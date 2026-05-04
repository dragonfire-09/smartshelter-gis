from __future__ import annotations

import html
import os
from typing import Any

import folium
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster


TURKEY_CENTER = [39.0, 35.0]
DEFAULT_ZOOM = 6

RISK_COLORS = {
    "Düşük": "#22c55e",
    "Orta": "#f59e0b",
    "Yüksek": "#ef4444",
    "Kritik": "#991b1b",
    "Veri yetersiz": "#94a3b8",
    "": "#94a3b8",
}


def _get_stadia_api_key() -> str:
    try:
        import streamlit as st

        key = st.secrets.get("STADIA_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.getenv("STADIA_API_KEY", "").strip()


STADIA_API_KEY = _get_stadia_api_key()


def _safe_float(val: Any, default: float | None = None) -> float | None:
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def _safe_str(val: Any, default: str = "—") -> str:
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return default

    return s


def _html_escape(val: Any, default: str = "—") -> str:
    return html.escape(_safe_str(val, default))


def _compute_center(df: pd.DataFrame | None):
    if df is None or df.empty:
        return TURKEY_CENTER, DEFAULT_ZOOM

    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "lon" if "lon" in df.columns else "longitude"

    if lat_col not in df.columns or lon_col not in df.columns:
        return TURKEY_CENTER, DEFAULT_ZOOM

    lat_s = pd.to_numeric(df[lat_col], errors="coerce")
    lon_s = pd.to_numeric(df[lon_col], errors="coerce")

    valid = (
        lat_s.notna()
        & lon_s.notna()
        & lat_s.between(-90, 90)
        & lon_s.between(-180, 180)
    )

    if not valid.any():
        return TURKEY_CENTER, DEFAULT_ZOOM

    center = [
        float(lat_s[valid].mean()),
        float(lon_s[valid].mean()),
    ]

    lat_range = float(lat_s[valid].max() - lat_s[valid].min())
    lon_range = float(lon_s[valid].max() - lon_s[valid].min())
    max_range = max(lat_range, lon_range)

    if max_range < 0.15:
        zoom = 12
    elif max_range < 0.5:
        zoom = 10
    elif max_range < 1.5:
        zoom = 9
    elif max_range < 4:
        zoom = 7
    else:
        zoom = DEFAULT_ZOOM

    return center, zoom


def _is_default_tile(layer_name: str, default_tile: str | None) -> bool:
    if not default_tile:
        return False

    a = layer_name.lower().strip()
    b = default_tile.lower().strip()

    return a == b or b in a or a in b


def _stadia_url(style: str) -> str:
    base_url = (
        f"https://tiles.stadiamaps.com/tiles/{style}"
        f"/{{z}}/{{x}}/{{y}}{{r}}.png"
    )

    if STADIA_API_KEY:
        return f"{base_url}?api_key={STADIA_API_KEY}"

    return base_url


def _add_tile_layers(
    m: folium.Map,
    default_tile: str = "Stadia OSM Bright",
):
    default_tile = default_tile or "Stadia OSM Bright"

    stadia_attr = (
        '&copy; <a href="https://www.stadiamaps.com/">Stadia Maps</a> '
        '&copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> '
        '&copy; <a href="https://www.openstreetmap.org/copyright">'
        "OpenStreetMap contributors</a>"
    )

    # Stadia Maps
    folium.TileLayer(
        tiles=_stadia_url("osm_bright"),
        name="🧭 Stadia OSM Bright",
        attr=stadia_attr,
        max_zoom=20,
        control=True,
        overlay=False,
        show=_is_default_tile("Stadia OSM Bright", default_tile),
    ).add_to(m)

    folium.TileLayer(
        tiles=_stadia_url("outdoors"),
        name="🏞️ Stadia Outdoors",
        attr=stadia_attr,
        max_zoom=20,
        control=True,
        overlay=False,
        show=_is_default_tile("Stadia Outdoors", default_tile),
    ).add_to(m)

    folium.TileLayer(
        tiles=_stadia_url("alidade_smooth"),
        name="☀️ Stadia Alidade Smooth",
        attr=stadia_attr,
        max_zoom=20,
        control=True,
        overlay=False,
        show=_is_default_tile("Stadia Alidade Smooth", default_tile),
    ).add_to(m)

    folium.TileLayer(
        tiles=_stadia_url("alidade_smooth_dark"),
        name="🌙 Stadia Alidade Dark",
        attr=stadia_attr,
        max_zoom=20,
        control=True,
        overlay=False,
        show=_is_default_tile("Stadia Alidade Dark", default_tile),
    ).add_to(m)

    folium.TileLayer(
        tiles=_stadia_url("alpenglow"),
        name="🌇 Stadia Alpenglow",
        attr=stadia_attr,
        max_zoom=20,
        control=True,
        overlay=False,
        show=_is_default_tile("Stadia Alpenglow", default_tile),
    ).add_to(m)

    # OpenStreetMap / CartoDB
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="🗺️ OpenStreetMap",
        attr="© OpenStreetMap contributors",
        max_zoom=19,
        control=True,
        overlay=False,
        show=_is_default_tile("OpenStreetMap", default_tile),
    ).add_to(m)

    folium.TileLayer(
        tiles="CartoDB positron",
        name="☀️ Açık Tema",
        attr="© OpenStreetMap © CARTO",
        max_zoom=20,
        control=True,
        overlay=False,
        show=(
            _is_default_tile("Açık Tema", default_tile)
            or _is_default_tile("CartoDB Positron", default_tile)
        ),
    ).add_to(m)

    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="🌙 Koyu Tema",
        attr="© OpenStreetMap © CARTO",
        max_zoom=20,
        control=True,
        overlay=False,
        show=(
            _is_default_tile("Koyu Tema", default_tile)
            or _is_default_tile("CartoDB Dark Matter", default_tile)
        ),
    ).add_to(m)

    # Esri
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        name="🛰️ Uydu Görüntüsü",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
        max_zoom=19,
        control=True,
        overlay=False,
        show=(
            _is_default_tile("Uydu Görüntüsü", default_tile)
            or _is_default_tile("Esri World Imagery", default_tile)
        ),
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        name="🏙️ Sokak Haritası",
        attr="Tiles © Esri",
        max_zoom=19,
        control=True,
        overlay=False,
        show=(
            _is_default_tile("Sokak Haritası", default_tile)
            or _is_default_tile("Esri Street", default_tile)
        ),
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        name="⛰️ Topografik",
        attr="Tiles © Esri",
        max_zoom=19,
        control=True,
        overlay=False,
        show=(
            _is_default_tile("Topografik", default_tile)
            or _is_default_tile("Esri Topo", default_tile)
        ),
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
        ),
        name="🌐 Uydu Etiketleri / Hibrit",
        attr="Tiles © Esri",
        max_zoom=19,
        control=True,
        overlay=True,
        show=False,
    ).add_to(m)


def _build_popup(row: pd.Series) -> str:
    name = _html_escape(row.get("name"))
    city = _html_escape(row.get("city"))
    district = _html_escape(row.get("district"))
    risk_level = _html_escape(row.get("risk_level"))
    portal = _html_escape(row.get("source_portal"))

    capacity = row.get("capacity")
    occupancy = row.get("occupancy")
    risk_score = row.get("risk_score")

    cap_txt = f"{int(capacity):,}" if pd.notna(capacity) else "—"
    occ_txt = f"{int(occupancy):,}" if pd.notna(occupancy) else "—"
    risk_txt = f"{float(risk_score):.1f}" if pd.notna(risk_score) else "—"

    raw_risk_level = _safe_str(row.get("risk_level"), "Veri yetersiz")
    risk_color = RISK_COLORS.get(raw_risk_level, "#666")

    return f"""
    <div style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        min-width: 240px;
        max-width: 320px;
    ">
        <h4 style="margin: 0 0 8px 0; color: #1e3a8a;">🏥 {name}</h4>
        <p style="margin: 4px 0;"><b>📍 Konum:</b> {city} / {district}</p>
        <p style="margin: 4px 0;"><b>🏠 Kapasite:</b> {cap_txt}</p>
        <p style="margin: 4px 0;"><b>🐾 Mevcut Hayvan:</b> {occ_txt}</p>
        <p style="margin: 4px 0;">
            <b>⚠️ Risk Skoru:</b> {risk_txt}
            <span style="color:{risk_color}; font-weight: 600;">({risk_level})</span>
        </p>
        <p style="margin: 6px 0 0 0; font-size: 11px; color: #666;">
            <b>📊 Kaynak:</b> {portal}
        </p>
    </div>
    """


def _marker_radius(capacity: Any, occupancy: Any) -> int:
    radius = 6

    if pd.notna(capacity) and pd.notna(occupancy):
        try:
            cap_f = float(capacity)
            occ_f = float(occupancy)
            if cap_f > 0:
                ratio = min(occ_f / cap_f, 2.0)
                radius = 5 + int(ratio * 8)
        except Exception:
            pass

    return radius


def _heat_weight(risk_score: Any) -> float:
    if pd.notna(risk_score):
        try:
            return max(float(risk_score) / 100.0, 0.3)
        except Exception:
            pass

    return 1.0


def _add_marker_layers(m: folium.Map, df: pd.DataFrame | None):
    if df is None or df.empty:
        return

    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "lon" if "lon" in df.columns else "longitude"

    if lat_col not in df.columns or lon_col not in df.columns:
        return

    all_markers = folium.FeatureGroup(name="📍 Tüm Barınaklar", show=True)
    cluster_group = MarkerCluster(name="🔵 Cluster / Yoğunluk", show=False)
    critical_markers = folium.FeatureGroup(name="🚨 Kritik Riskli", show=True)
    heatmap_data: list[list[float]] = []

    for _, row in df.iterrows():
        lat = _safe_float(row.get(lat_col))
        lon = _safe_float(row.get(lon_col))

        if lat is None or lon is None:
            continue

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            continue

        risk_level = _safe_str(row.get("risk_level"), "Veri yetersiz")
        color = RISK_COLORS.get(risk_level, "#94a3b8")

        capacity = row.get("capacity")
        occupancy = row.get("occupancy")
        radius = _marker_radius(capacity, occupancy)

        popup_html = _build_popup(row)
        tooltip = _html_escape(row.get("name"))

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.72,
            weight=2,
            popup=folium.Popup(popup_html, max_width=340),
            tooltip=tooltip,
        ).add_to(all_markers)

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.72,
            weight=2,
            popup=folium.Popup(popup_html, max_width=340),
            tooltip=tooltip,
        ).add_to(cluster_group)

        if risk_level == "Kritik":
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="red", icon="warning-sign"),
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"🚨 KRİTİK: {tooltip}",
            ).add_to(critical_markers)

        heatmap_data.append(
            [
                float(lat),
                float(lon),
                float(_heat_weight(row.get("risk_score"))),
            ]
        )

    all_markers.add_to(m)
    critical_markers.add_to(m)
    cluster_group.add_to(m)

    if heatmap_data:
        heatmap_layer = folium.FeatureGroup(
            name="🔥 Risk Yoğunluk Haritası",
            show=False,
        )

        HeatMap(
            heatmap_data,
            min_opacity=0.3,
            radius=20,
            blur=15,
            max_zoom=12,
        ).add_to(heatmap_layer)

        heatmap_layer.add_to(m)


def _add_leaflet_control_css(m: folium.Map):
    css = """
    <style>
        .leaflet-top.leaflet-right {
            z-index: 999999 !important;
        }

        .leaflet-control-layers {
            z-index: 999999 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
            font-size: 13px !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 14px rgba(0,0,0,0.22) !important;
        }

        .leaflet-control-layers-expanded {
            padding: 10px 12px !important;
            background: white !important;
            color: #111827 !important;
            max-height: 460px !important;
            overflow-y: auto !important;
        }

        .leaflet-control-layers label {
            margin-bottom: 5px !important;
            cursor: pointer !important;
            white-space: nowrap !important;
        }

        .leaflet-control-layers-selector {
            margin-right: 6px !important;
        }

        .leaflet-control-container {
            z-index: 999999 !important;
        }
    </style>
    """
    m.get_root().header.add_child(folium.Element(css))


def _add_legend(m: folium.Map):
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
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 13px;
        line-height: 1.45;
    ">
        <b style="font-size: 14px;">Risk Seviyesi</b><br>
        <span style="color:#22c55e;">●</span> Düşük<br>
        <span style="color:#f59e0b;">●</span> Orta<br>
        <span style="color:#ef4444;">●</span> Yüksek<br>
        <span style="color:#991b1b;">●</span> Kritik<br>
        <span style="color:#94a3b8;">●</span> Veri yetersiz<br>
        <span style="display:block; color:#666; font-size: 11px; margin-top: 6px;">
            Marker boyutu doluluk oranına göre değişir.
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def _add_fullscreen(m: folium.Map):
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


def create_shelter_map(
    df: pd.DataFrame | None,
    default_tile: str = "Stadia OSM Bright",
) -> folium.Map:
    center, zoom = _compute_center(df)

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    _add_tile_layers(m, default_tile=default_tile)
    _add_marker_layers(m, df)

    folium.LayerControl(
        position="topright",
        collapsed=False,
        autoZIndex=True,
    ).add_to(m)

    _add_leaflet_control_css(m)
    _add_legend(m)
    _add_fullscreen(m)

    return m
