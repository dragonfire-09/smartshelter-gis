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
