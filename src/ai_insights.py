import pandas as pd


def calculate_data_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df.empty:
        df["data_quality_score"] = []
        df["data_quality_level"] = []
        return df

    score = pd.Series(0, index=df.index, dtype=float)

    score += df["name"].astype(str).str.strip().ne("").astype(int) * 10
    score += df["city"].astype(str).str.strip().ne("Belirtilmemiş").astype(int) * 10
    score += df["district"].astype(str).str.strip().ne("Belirtilmemiş").astype(int) * 10

    if "coordinate_valid" in df.columns:
        score += df["coordinate_valid"].astype(bool).astype(int) * 20

    if "capacity_available" in df.columns:
        score += df["capacity_available"].astype(bool).astype(int) * 20
    elif "capacity_estimated" in df.columns:
        score += (~df["capacity_estimated"].astype(bool)).astype(int) * 20

    if "occupancy_available" in df.columns:
        score += df["occupancy_available"].astype(bool).astype(int) * 20
    elif "occupancy_estimated" in df.columns:
        score += (~df["occupancy_estimated"].astype(bool)).astype(int) * 20

    if "vet_count_available" in df.columns:
        score += df["vet_count_available"].astype(bool).astype(int) * 10
    elif "vet_count_estimated" in df.columns:
        score += (~df["vet_count_estimated"].astype(bool)).astype(int) * 10

    if "sterilization_available" in df.columns:
        score += df["sterilization_available"].astype(bool).astype(int) * 5

    if "adoption_available" in df.columns:
        score += df["adoption_available"].astype(bool).astype(int) * 5

    df["data_quality_score"] = score.clip(0, 100).round(0).astype(int)

    df["data_quality_level"] = pd.cut(
        df["data_quality_score"],
        bins=[-1, 49, 74, 100],
        labels=["Düşük", "Orta", "Yüksek"],
    )

    return df


def generate_risk_explanations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df.empty:
        df["risk_explanation"] = []
        return df

    explanations = []

    for _, row in df.iterrows():
        if not bool(row.get("risk_eligible", True)):
            scope = row.get("data_scope", "")

            if scope == "capacity_only":
                explanations.append(
                    "Bu kayıt kapasite bilgisi içeriyor ancak mevcut hayvan sayısı bulunmadığı için risk skoru hesaplanmamıştır. Doluluk ve risk analizi için mevcut hayvan sayısı gereklidir."
                )
            elif scope == "location_only":
                explanations.append(
                    "Bu kayıt konum veya tesis bilgisi içeriyor ancak kapasite ve mevcut hayvan sayısı bulunmadığı için risk analizi yapılamaz."
                )
            else:
                explanations.append(
                    "Bu kayıt ana risk analizi için yeterli veri içermiyor. Kaynak veri ayrıca incelenmelidir."
                )

            continue

        parts = []

        occupancy_rate = row.get("occupancy_rate", None)
        animals_per_vet = row.get("animals_per_vet", None)
        risk_score = row.get("risk_score", None)
        risk_level = str(row.get("risk_level", ""))

        if pd.notna(occupancy_rate):
            if occupancy_rate >= 100:
                parts.append(
                    f"Doluluk oranı %{occupancy_rate:.1f} ile kapasite sınırının üzerindedir."
                )
            elif occupancy_rate >= 85:
                parts.append(
                    f"Doluluk oranı %{occupancy_rate:.1f} seviyesinde ve yakından izlenmelidir."
                )
            else:
                parts.append(
                    f"Doluluk oranı %{occupancy_rate:.1f} seviyesinde görünmektedir."
                )

        if pd.notna(animals_per_vet):
            if animals_per_vet >= 150:
                parts.append(
                    f"Veteriner başına düşen hayvan sayısı {animals_per_vet:.1f}; bu yüksek iş yüküne işaret eder."
                )
            elif animals_per_vet >= 100:
                parts.append(
                    f"Veteriner başına düşen hayvan sayısı {animals_per_vet:.1f}; orta düzey operasyonel baskı vardır."
                )
            else:
                parts.append(
                    f"Veteriner başına düşen hayvan sayısı {animals_per_vet:.1f}; göreli olarak yönetilebilir düzeydedir."
                )

        occupancy = max(row.get("occupancy", 0), 1)

        if row.get("adoption_available", False):
            adoption_ratio = row.get("adoption_count", 0) / occupancy

            if adoption_ratio < 0.10:
                parts.append(
                    "Sahiplendirme sayısı mevcut hayvan sayısına göre düşük görünmektedir."
                )
            elif adoption_ratio < 0.25:
                parts.append("Sahiplendirme performansı orta düzeydedir.")
            else:
                parts.append("Sahiplendirme performansı olumlu görünmektedir.")

        if row.get("sterilization_available", False):
            sterilization_ratio = row.get("sterilization_count", 0) / occupancy

            if sterilization_ratio < 0.20:
                parts.append(
                    "Kısırlaştırma sayısı mevcut hayvan sayısına göre düşük olabilir."
                )

        if row.get("is_estimated", False):
            parts.append(
                "Bazı alanlar eksik olduğu için kaynak veri doğrulaması önerilir."
            )

        if pd.notna(risk_score):
            if risk_level == "Kritik":
                conclusion = (
                    f"Genel risk skoru {risk_score:.1f}. Kayıt kritik öncelikte değerlendirilmelidir."
                )
            elif risk_level == "Orta":
                conclusion = (
                    f"Genel risk skoru {risk_score:.1f}. Kayıt orta öncelikte izlenmelidir."
                )
            else:
                conclusion = (
                    f"Genel risk skoru {risk_score:.1f}. Kayıt düşük risk grubundadır."
                )
        else:
            conclusion = "Risk skoru hesaplanamamıştır."

        explanations.append(" ".join(parts + [conclusion]))

    df["risk_explanation"] = explanations

    return df


def generate_executive_summary(
    df: pd.DataFrame,
    history_summary_df: pd.DataFrame | None = None,
    anomalies_df: pd.DataFrame | None = None,
) -> str:
    if df.empty:
        return "Veri setinde analiz edilecek kayıt bulunamadı."

    risk_df = (
        df[df["risk_eligible"].astype(bool)].copy()
        if "risk_eligible" in df.columns
        else df.copy()
    )

    capacity_df = (
        df[df["capacity_available"].astype(bool)].copy()
        if "capacity_available" in df.columns
        else df.copy()
    )

    occupancy_df = (
        df[df["occupancy_available"].astype(bool)].copy()
        if "occupancy_available" in df.columns
        else df.copy()
    )

    total_records = len(df)
    risk_ready_count = len(risk_df)
    capacity_record_count = len(capacity_df)

    total_capacity = int(capacity_df["capacity"].sum()) if len(capacity_df) else 0
    total_occupancy = int(occupancy_df["occupancy"].sum()) if len(occupancy_df) else 0

    avg_risk = risk_df["risk_score"].mean() if len(risk_df) else 0

    avg_quality = (
        df["data_quality_score"].mean()
        if "data_quality_score" in df.columns
        else 0
    )

    critical_count = len(risk_df[risk_df["risk_level"].astype(str) == "Kritik"])
    medium_count = len(risk_df[risk_df["risk_level"].astype(str) == "Orta"])
    low_count = len(risk_df[risk_df["risk_level"].astype(str) == "Düşük"])

    summary = f"""
### AI Yönetici Özeti

Bu veri setinde toplam **{total_records}** envanter kaydı bulunmaktadır. Bunların **{risk_ready_count}** tanesi risk analizi için yeterli kapasite ve mevcut hayvan verisine sahiptir. **{capacity_record_count}** kayıtta kapasite bilgisi bulunmaktadır.

Bilinen toplam kapasite **{total_capacity}**, bilinen mevcut hayvan sayısı **{total_occupancy}** olarak hesaplanmıştır. Ortalama veri kalitesi skoru **{avg_quality:.1f}/100** seviyesindedir.
"""

    if len(risk_df):
        summary += f"""
Risk hesaplanabilir kayıtlar içinde **{critical_count} kritik**, **{medium_count} orta**, **{low_count} düşük** riskli kayıt bulunmaktadır. Ortalama risk skoru **{avg_risk:.1f}** seviyesindedir.

#### Öncelikli Riskli Kayıtlar
"""

        top_risk = risk_df.sort_values("risk_score", ascending=False).head(5)

        for _, row in top_risk.iterrows():
            summary += (
                f"- **{row['name']}** / {row['district']} - "
                f"Risk: **{row['risk_score']}**, "
                f"Doluluk: **%{row['occupancy_rate']}**, "
                f"Veteriner başına hayvan: **{row['animals_per_vet']}**\n"
            )

        worst_districts = (
            risk_df.groupby("district", as_index=False)
            .agg(
                avg_risk=("risk_score", "mean"),
                center_count=("name", "count"),
                total_occupancy=("occupancy", "sum"),
            )
            .sort_values("avg_risk", ascending=False)
            .head(3)
        )

        summary += "\n#### Risk Açısından Öne Çıkan İlçeler\n"

        for _, row in worst_districts.iterrows():
            summary += (
                f"- **{row['district']}**: Ortalama risk **{row['avg_risk']:.1f}**, "
                f"kayıt sayısı **{int(row['center_count'])}**, "
                f"mevcut hayvan **{int(row['total_occupancy'])}**\n"
            )

    else:
        summary += """
Risk hesaplanabilir kayıt bulunmamaktadır. Risk analizi için kapasite ve mevcut hayvan sayısının birlikte sağlanması gerekir.
"""

    if anomalies_df is not None and not anomalies_df.empty:
        high_anomalies = anomalies_df[
            anomalies_df["severity"].isin(["Yüksek", "Kritik"])
        ]

        summary += "\n#### AI Uyarıları\n"

        if len(high_anomalies) > 0:
            summary += (
                f"Sistemde **{len(high_anomalies)}** yüksek/kritik seviyede anomali tespit edilmiştir.\n"
            )
        else:
            summary += "Yüksek veya kritik seviyede anomali tespit edilmemiştir.\n"

    if history_summary_df is not None and len(history_summary_df) >= 2:
        old = history_summary_df.iloc[-2]
        new = history_summary_df.iloc[-1]

        occupancy_delta = new["total_occupancy"] - old["total_occupancy"]
        risk_delta = new["avg_risk"] - old["avg_risk"]

        summary += "\n#### Son Snapshot Değişimi\n"
        summary += (
            f"Son iki snapshot arasında mevcut hayvan sayısı **{occupancy_delta:+.0f}**, "
            f"ortalama risk skoru **{risk_delta:+.1f}** değişmiştir.\n"
        )

    summary += """
#### Genel Öneriler

1. Risk analizi için kapasite ve mevcut hayvan sayısı birlikte toplanmalıdır.  
2. Sadece kapasite içeren kayıtlar doluluk verisiyle tamamlanmalıdır.  
3. Veri kalitesi düşük kaynaklar için belediye/kaynak doğrulaması yapılmalıdır.  
4. Operasyonel istatistik kaynakları risk dashboard’una karıştırılmadan ayrı analiz edilmelidir.
"""

    return summary


def detect_anomalies(history_df: pd.DataFrame, current_df: pd.DataFrame | None = None) -> pd.DataFrame:
    anomalies = []

    if current_df is not None and not current_df.empty:
        for _, row in current_df.iterrows():
            name = row.get("name", "")
            city = row.get("city", "")
            district = row.get("district", "")

            if bool(row.get("risk_eligible", False)):
                if row.get("occupancy_rate", 0) >= 120:
                    anomalies.append(
                        {
                            "severity": "Kritik",
                            "type": "Kapasite aşımı",
                            "name": name,
                            "city": city,
                            "district": district,
                            "message": f"Doluluk oranı %{row.get('occupancy_rate', 0):.1f}. Kapasite ciddi şekilde aşılmış görünüyor.",
                        }
                    )

                if row.get("vet_count_available", False) and row.get("vet_count", 0) == 0:
                    anomalies.append(
                        {
                            "severity": "Yüksek",
                            "type": "Veteriner eksikliği",
                            "name": name,
                            "city": city,
                            "district": district,
                            "message": "Veteriner sayısı 0 görünüyor.",
                        }
                    )

            if not row.get("coordinate_valid", True):
                anomalies.append(
                    {
                        "severity": "Orta",
                        "type": "Koordinat sorunu",
                        "name": name,
                        "city": city,
                        "district": district,
                        "message": "Koordinat eksik veya Türkiye sınırları dışında görünüyor.",
                    }
                )

            if row.get("data_quality_score", 100) < 50:
                anomalies.append(
                    {
                        "severity": "Orta",
                        "type": "Düşük veri kalitesi",
                        "name": name,
                        "city": city,
                        "district": district,
                        "message": f"Veri kalite skoru {row.get('data_quality_score', 0)}/100. Kaynak veri doğrulaması önerilir.",
                    }
                )

    if not anomalies:
        return pd.DataFrame(
            columns=["severity", "type", "name", "city", "district", "message"]
        )

    anomalies_df = pd.DataFrame(anomalies)

    severity_order = {
        "Kritik": 0,
        "Yüksek": 1,
        "Orta": 2,
        "Düşük": 3,
    }

    anomalies_df["severity_order"] = anomalies_df["severity"].map(severity_order)
    anomalies_df = anomalies_df.sort_values(["severity_order", "type", "name"])
    anomalies_df = anomalies_df.drop(columns=["severity_order"])

    return anomalies_df


def simulate_interventions(
    df: pd.DataFrame,
    extra_capacity: int = 0,
    extra_vets: int = 0,
    extra_adoptions: int = 0,
    extra_sterilizations: int = 0,
) -> pd.DataFrame:
    from src.risk import calculate_risk

    if df.empty:
        return pd.DataFrame()

    if "risk_eligible" in df.columns:
        scenario_df = df[df["risk_eligible"].astype(bool)].copy()
    else:
        scenario_df = df.copy()

    if scenario_df.empty:
        return pd.DataFrame()

    scenario_df["base_capacity"] = scenario_df["capacity"]
    scenario_df["base_vet_count"] = scenario_df["vet_count"]
    scenario_df["base_adoption_count"] = scenario_df["adoption_count"]
    scenario_df["base_sterilization_count"] = scenario_df["sterilization_count"]
    scenario_df["base_risk_score"] = scenario_df["risk_score"]
    scenario_df["base_risk_level"] = scenario_df["risk_level"]
    scenario_df["base_occupancy_rate"] = scenario_df["occupancy_rate"]

    scenario_df["capacity"] = scenario_df["capacity"] + extra_capacity
    scenario_df["vet_count"] = scenario_df["vet_count"] + extra_vets
    scenario_df["adoption_count"] = scenario_df["adoption_count"] + extra_adoptions
    scenario_df["sterilization_count"] = scenario_df["sterilization_count"] + extra_sterilizations

    scenario_df = calculate_risk(scenario_df)

    scenario_df["scenario_risk_score"] = scenario_df["risk_score"]
    scenario_df["scenario_risk_level"] = scenario_df["risk_level"]
    scenario_df["scenario_occupancy_rate"] = scenario_df["occupancy_rate"]

    scenario_df["risk_score_delta"] = (
        scenario_df["scenario_risk_score"] - scenario_df["base_risk_score"]
    ).round(1)

    scenario_df["risk_score_improvement"] = (
        scenario_df["base_risk_score"] - scenario_df["scenario_risk_score"]
    ).round(1)

    output_cols = [
        "name",
        "city",
        "district",
        "base_risk_score",
        "scenario_risk_score",
        "risk_score_delta",
        "risk_score_improvement",
        "base_risk_level",
        "scenario_risk_level",
        "base_occupancy_rate",
        "scenario_occupancy_rate",
        "base_capacity",
        "capacity",
        "base_vet_count",
        "vet_count",
        "base_adoption_count",
        "adoption_count",
        "base_sterilization_count",
        "sterilization_count",
    ]

    output_cols = [c for c in output_cols if c in scenario_df.columns]

    return scenario_df[output_cols].sort_values(
        "risk_score_improvement",
        ascending=False,
    )
