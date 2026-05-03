import pandas as pd
import plotly.express as px


def empty_figure(title="Veri bulunamadı"):
    fig = px.scatter(title=title)

    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "Filtreye uygun veri bulunamadı.",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
    )

    return fig


def chart_risk_score(df):
    if df.empty:
        return empty_figure("Risk Skoru")

    plot_df = df.sort_values("risk_score", ascending=False).copy()

    fig = px.bar(
        plot_df,
        x="name",
        y="risk_score",
        color="risk_level",
        text="risk_score",
        title="Risk Skoru",
        color_discrete_map={
            "Düşük": "#16a34a",
            "Orta": "#f59e0b",
            "Kritik": "#dc2626",
        },
        hover_data=[
            "city",
            "district",
            "capacity",
            "occupancy",
            "occupancy_rate",
            "animals_per_vet",
        ],
    )

    fig.update_layout(
        xaxis_title="Kayıt",
        yaxis_title="Risk Skoru",
        xaxis_tickangle=-35,
        legend_title="Risk Seviyesi",
    )

    fig.update_traces(textposition="outside")

    return fig


def chart_occupancy_rate(df):
    if df.empty:
        return empty_figure("Doluluk Oranı")

    plot_df = df.sort_values("occupancy_rate", ascending=False).copy()

    fig = px.bar(
        plot_df,
        x="name",
        y="occupancy_rate",
        color="district",
        text="occupancy_rate",
        title="Doluluk Oranı (%)",
        hover_data=[
            "city",
            "district",
            "capacity",
            "occupancy",
            "risk_level",
            "risk_score",
        ],
    )

    fig.update_layout(
        xaxis_title="Kayıt",
        yaxis_title="Doluluk Oranı (%)",
        xaxis_tickangle=-35,
        legend_title="İlçe",
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    return fig


def build_district_summary(df):
    if df.empty:
        return pd.DataFrame(
            columns=[
                "district",
                "center_count",
                "total_capacity",
                "total_occupancy",
                "occupancy_rate",
                "avg_risk",
                "critical_count",
                "estimated_record_count",
            ]
        )

    summary = (
        df.groupby("district", as_index=False)
        .agg(
            center_count=("name", "count"),
            total_capacity=("capacity", "sum"),
            total_occupancy=("occupancy", "sum"),
            avg_risk=("risk_score", "mean"),
            critical_count=(
                "risk_level",
                lambda s: (s.astype(str) == "Kritik").sum(),
            ),
            estimated_record_count=(
                "is_estimated",
                lambda s: (s == True).sum(),  # noqa: E712
            ),
        )
    )

    summary["occupancy_rate"] = (
        summary["total_occupancy"]
        / summary["total_capacity"].replace(0, 1)
        * 100
    ).round(1)

    summary["avg_risk"] = summary["avg_risk"].round(1)

    summary = summary.sort_values("avg_risk", ascending=False)

    return summary


def chart_district_avg_risk(summary_df):
    if summary_df.empty:
        return empty_figure("İlçe Bazlı Ortalama Risk")

    fig = px.bar(
        summary_df,
        x="district",
        y="avg_risk",
        color="avg_risk",
        text="avg_risk",
        title="İlçe Bazlı Ortalama Risk",
        color_continuous_scale="RdYlGn_r",
        hover_data=[
            "center_count",
            "total_capacity",
            "total_occupancy",
            "occupancy_rate",
            "critical_count",
        ],
    )

    fig.update_layout(
        xaxis_title="İlçe",
        yaxis_title="Ortalama Risk",
        coloraxis_colorbar_title="Risk",
    )

    fig.update_traces(textposition="outside")

    return fig


def chart_history_trend(summary_df):
    if summary_df.empty:
        return empty_figure("Tarihsel Trend")

    long_df = summary_df.melt(
        id_vars=["snapshot_date"],
        value_vars=[
            "record_count",
            "total_capacity",
            "total_occupancy",
            "avg_risk",
            "critical_count",
        ],
        var_name="metric",
        value_name="value",
    )

    metric_names = {
        "record_count": "Kayıt Sayısı",
        "total_capacity": "Toplam Kapasite",
        "total_occupancy": "Mevcut Hayvan",
        "avg_risk": "Ortalama Risk",
        "critical_count": "Kritik Kayıt",
    }

    long_df["metric_label"] = long_df["metric"].map(metric_names)

    fig = px.line(
        long_df,
        x="snapshot_date",
        y="value",
        color="metric_label",
        markers=True,
        title="Tarihsel Göstergeler",
    )

    fig.update_layout(
        xaxis_title="Tarih",
        yaxis_title="Değer",
        legend_title="Gösterge",
    )

    return fig


def chart_history_metric(summary_df, metric):
    if summary_df.empty or metric not in summary_df.columns:
        return empty_figure("Tarihsel Metrik")

    metric_titles = {
        "record_count": "Kayıt Sayısı",
        "total_capacity": "Toplam Kapasite",
        "total_occupancy": "Mevcut Hayvan",
        "avg_risk": "Ortalama Risk",
        "critical_count": "Kritik Kayıt",
        "estimated_count": "Tahmini Veri İçeren Kayıt",
    }

    fig = px.bar(
        summary_df,
        x="snapshot_date",
        y=metric,
        text=metric,
        title=f"Tarihsel Değişim: {metric_titles.get(metric, metric)}",
    )

    fig.update_layout(
        xaxis_title="Tarih",
        yaxis_title=metric_titles.get(metric, metric),
    )

    fig.update_traces(textposition="outside")

    return fig


def chart_record_delta(compare_df, metric_delta="risk_score_delta"):
    if compare_df.empty or metric_delta not in compare_df.columns:
        return empty_figure("Merkez Bazlı Değişim")

    title_map = {
        "risk_score_delta": "Risk Skoru Değişimi",
        "occupancy_delta": "Mevcut Hayvan Sayısı Değişimi",
        "capacity_delta": "Kapasite Değişimi",
        "occupancy_rate_delta": "Doluluk Oranı Değişimi",
        "vet_count_delta": "Veteriner Sayısı Değişimi",
    }

    plot_df = compare_df.copy()
    plot_df = plot_df.sort_values(metric_delta, ascending=False)

    fig = px.bar(
        plot_df,
        x="name",
        y=metric_delta,
        color="change_status",
        text=metric_delta,
        title=title_map.get(metric_delta, metric_delta),
        hover_data=[
            "city",
            "district",
            "change_status",
            "risk_score_old",
            "risk_score_new",
            "occupancy_old",
            "occupancy_new",
        ],
    )

    fig.update_layout(
        xaxis_title="Merkez",
        yaxis_title="Değişim",
        xaxis_tickangle=-35,
        legend_title="Durum",
    )

    fig.update_traces(textposition="outside")

    return fig
