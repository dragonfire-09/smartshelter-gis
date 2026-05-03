import pandas as pd


def calculate_data_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her kayıt için 0-100 arası veri kalite skoru üretir.

    Skor mantığı:
    - İl/ilçe/ad bilgisi
    - Koordinat geçerliliği
    - Kapasite gerçek/tahmini durumu
    - Mevcut hayvan gerçek/tahmini durumu
    - Veteriner gerçek/tahmini durumu
    - Operasyonel alanların varlığı
    """

    df = df.copy()

    if df.empty:
        df["data_quality_score"] = []
        df["data_quality_level"] = []
        return df

    score = pd.Series(0, index=df.index, dtype=float)

    # Kimlik bilgileri
    score += df["name"].astype(str).str.strip().ne("").astype(int) * 10
    score += df["city"].astype(str).str.strip().ne("Belirtilmemiş").astype(int) * 10
    score += df["district"].astype(str).str.strip().ne("Belirtilmemiş").astype(int) * 10

    # Koordinat
    score += df["coordinate_valid"].astype(bool).astype(int) * 25

    # Temel operasyonel veriler
    score += (~df["capacity_estimated"].astype(bool)).astype(int) * 15
    score += (~df["occupancy_estimated"].astype(bool)).astype(int) * 15
    score += (~df["vet_count_estimated"].astype(bool)).astype(int) * 10

    # Kısırlaştırma / sahiplendirme alanları tahmini değilse küçük katkı
    if "sterilization_count_estimated" in df.columns:
        score += (~df["sterilization_count_estimated"].astype(bool)).astype(int) * 3

    if "adoption_count_estimated" in df.columns:
        score += (~df["adoption_count_estimated"].astype(bool)).astype(int) * 2

    df["data_quality_score"] = score.clip(0, 100).round(0).astype(int)

    df["data_quality_level"] = pd.cut(
        df["data_quality_score"],
        bins=[-1, 49, 74, 100],
        labels=["Düşük", "Orta", "Yüksek"],
    )

    return df


def generate_risk_explanations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her kayıt için doğal dile yakın risk açıklaması üretir.
    LLM kullanmaz, kural tabanlıdır.
    """

    df = df.copy()

    if df.empty:
        df["risk_explanation"] = []
        return df

    explanations = []

    for _, row in df.iterrows():
        parts = []

        occupancy_rate = row.get("occupancy_rate", 0)
        animals_per_vet = row.get("animals_per_vet", 0)
        risk_score = row.get("risk_score", 0)
        risk_level = str(row.get("risk_level", ""))

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

        adoption_count = row.get("adoption_count", 0)
        occupancy = max(row.get("occupancy", 0), 1)

        adoption_ratio = adoption_count / occupancy

        if adoption_ratio < 0.10:
            parts.append(
                "Sahiplendirme sayısı mevcut hayvan sayısına göre düşük görünmektedir."
            )
        elif adoption_ratio < 0.25:
            parts.append(
                "Sahiplendirme performansı orta düzeydedir."
            )
        else:
            parts.append(
                "Sahiplendirme performansı olumlu görünmektedir."
            )

        sterilization_count = row.get("sterilization_count", 0)
        sterilization_ratio = sterilization_count / occupancy

        if sterilization_ratio < 0.20:
            parts.append(
                "Kısırlaştırma sayısı mevcut hayvan sayısına göre düşük olabilir."
            )

        if row.get("is_estimated", False):
            parts.append(
                "Bazı alanlar tahmini değerle tamamlandığı için veri doğrulaması önerilir."
            )

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

        explanation = " ".join(parts + [conclusion])
        explanations.append(explanation)

    df["risk_explanation"] = explanations

    return df


def generate_executive_summary(
    df: pd.DataFrame,
    history_summary_df: pd.DataFrame | None = None,
    anomalies_df: pd.DataFrame | None = None,
) -> str:
    """
    Dashboard için yönetici özeti üretir.
    LLM kullanmaz, kural tabanlıdır.
    """

    if df.empty:
        return "Veri setinde analiz edilecek kayıt bulunamadı."

    total_records = len(df)
    total_capacity = int(df["capacity"].sum())
    total_occupancy = int(df["occupancy"].sum())
    avg_risk = df["risk_score"].mean()
    avg_occupancy_rate = (
        total_occupancy / total_capacity * 100
        if total_capacity > 0
        else 0
    )

    critical_count = len(df[df["risk_level"].astype(str) == "Kritik"])
    medium_count = len(df[df["risk_level"].astype(str) == "Orta"])
    low_count = len(df[df["risk_level"].astype(str) == "Düşük"])

    avg_quality = (
        df["data_quality_score"].mean()
        if "data_quality_score" in df.columns
        else 0
    )

    top_risk = df.sort_values("risk_score", ascending=False).head(5)

    worst_districts = (
        df.groupby("district", as_index=False)
        .agg(
            avg_risk=("risk_score", "mean"),
            center_count=("name", "count"),
            total_occupancy=("occupancy", "sum"),
        )
        .sort_values("avg_risk", ascending=False)
        .head(3)
    )

    summary = f"""
### AI Yönetici Özeti

Bu veri setinde toplam **{total_records}** kayıt bulunmaktadır. Toplam kapasite **{total_capacity}**, mevcut hayvan sayısı **{total_occupancy}** olarak hesaplanmıştır. Genel doluluk oranı yaklaşık **%{avg_occupancy_rate:.1f}**, ortalama risk skoru ise **{avg_risk:.1f}** seviyesindedir.

Risk dağılımına göre **{critical_count} kritik**, **{medium_count} orta**, **{low_count} düşük** riskli kayıt bulunmaktadır. Ortalama veri kalitesi skoru **{avg_quality:.1f}/100** seviyesindedir.

#### Öncelikli Riskli Kayıtlar
"""

    for _, row in top_risk.iterrows():
        summary += (
            f"- **{row['name']}** / {row['district']} - "
            f"Risk: **{row['risk_score']}**, "
            f"Doluluk: **%{row['occupancy_rate']}**, "
            f"Veteriner başına hayvan: **{row['animals_per_vet']}**\n"
        )

    summary += "\n#### Risk Açısından Öne Çıkan İlçeler\n"

    for _, row in worst_districts.iterrows():
        summary += (
            f"- **{row['district']}**: Ortalama risk **{row['avg_risk']:.1f}**, "
            f"kayıt sayısı **{int(row['center_count'])}**, "
            f"mevcut hayvan **{int(row['total_occupancy'])}**\n"
        )

    if anomalies_df is not None and not anomalies_df.empty:
        high_anomalies = anomalies_df[
            anomalies_df["severity"].isin(["Yüksek", "Kritik"])
        ]

        summary += "\n#### AI Uyarıları\n"

        if len(high_anomalies) > 0:
            summary += (
                f"Sistemde **{len(high_anomalies)}** yüksek/kritik seviyede anomali tespit edilmiştir. "
                "Bu kayıtların veri doğrulaması ve operasyonel incelemesi önerilir.\n"
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

1. Kritik ve orta riskli merkezlerde kapasite ve veteriner yükü birlikte değerlendirilmelidir.  
2. Veri kalitesi düşük kayıtlar için belediye/kaynak veri doğrulaması yapılmalıdır.  
3. Sahiplendirme ve kısırlaştırma performansı düşük merkezlerde hedefli operasyon planı hazırlanmalıdır.  
4. Geçmiş snapshot verileri biriktikçe risk trendi ve erken uyarı analizleri daha güvenilir hale gelecektir.
"""

    return summary


def detect_anomalies(
    history_df: pd.DataFrame,
    current_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Tarihsel snapshot ve güncel veri üzerinden anomali tespiti yapar.
    LLM kullanmaz, kural tabanlıdır.
    """

    anomalies = []

    # Güncel veri üzerinden statik anomali kontrolleri
    if current_df is not None and not current_df.empty:
        for _, row in current_df.iterrows():
            name = row.get("name", "")
            city = row.get("city", "")
            district = row.get("district", "")

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

            if row.get("vet_count", 0) == 0:
                anomalies.append(
                    {
                        "severity": "Yüksek",
                        "type": "Veteriner eksikliği",
                        "name": name,
                        "city": city,
                        "district": district,
                        "message": "Veteriner sayısı 0 görünüyor. Veri hatası veya ciddi personel eksikliği olabilir.",
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

    # Tarihsel karşılaştırma anomalileri
    if history_df is not None and not history_df.empty:
        dates = (
            history_df["snapshot_date"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        if len(dates) >= 2:
            prev_date = dates[-2]
            last_date = dates[-1]

            prev_df = history_df[
                history_df["snapshot_date"].astype(str) == prev_date
            ].copy()

            last_df = history_df[
                history_df["snapshot_date"].astype(str) == last_date
            ].copy()

            keep = [
                "record_key",
                "name",
                "city",
                "district",
                "capacity",
                "occupancy",
                "risk_score",
                "occupancy_rate",
            ]

            prev_df = prev_df[[c for c in keep if c in prev_df.columns]].add_suffix("_old")
            last_df = last_df[[c for c in keep if c in last_df.columns]].add_suffix("_new")

            merged = prev_df.merge(
                last_df,
                left_on="record_key_old",
                right_on="record_key_new",
                how="inner",
            )

            for _, row in merged.iterrows():
                name = row.get("name_new", row.get("name_old", ""))
                city = row.get("city_new", row.get("city_old", ""))
                district = row.get("district_new", row.get("district_old", ""))

                old_occ = row.get("occupancy_old", 0)
                new_occ = row.get("occupancy_new", 0)

                if old_occ and old_occ > 0:
                    occ_change_ratio = (new_occ - old_occ) / old_occ

                    if occ_change_ratio >= 0.30:
                        anomalies.append(
                            {
                                "severity": "Yüksek",
                                "type": "Ani hayvan sayısı artışı",
                                "name": name,
                                "city": city,
                                "district": district,
                                "message": f"Mevcut hayvan sayısı {prev_date} tarihinden {last_date} tarihine %{occ_change_ratio * 100:.1f} arttı.",
                            }
                        )

                    if occ_change_ratio <= -0.50:
                        anomalies.append(
                            {
                                "severity": "Orta",
                                "type": "Ani hayvan sayısı düşüşü",
                                "name": name,
                                "city": city,
                                "district": district,
                                "message": f"Mevcut hayvan sayısı {prev_date} tarihinden {last_date} tarihine %{abs(occ_change_ratio) * 100:.1f} düştü. Veri doğrulaması önerilir.",
                            }
                        )

                old_capacity = row.get("capacity_old", 0)
                new_capacity = row.get("capacity_new", 0)

                if old_capacity and old_capacity > 0:
                    cap_change_ratio = (new_capacity - old_capacity) / old_capacity

                    if abs(cap_change_ratio) >= 0.40:
                        anomalies.append(
                            {
                                "severity": "Orta",
                                "type": "Ani kapasite değişimi",
                                "name": name,
                                "city": city,
                                "district": district,
                                "message": f"Kapasite {prev_date} - {last_date} arasında %{cap_change_ratio * 100:.1f} değişti. Veri girişi kontrol edilmeli.",
                            }
                        )

                old_risk = row.get("risk_score_old", 0)
                new_risk = row.get("risk_score_new", 0)
                risk_delta = new_risk - old_risk

                if risk_delta >= 20:
                    anomalies.append(
                        {
                            "severity": "Yüksek",
                            "type": "Ani risk artışı",
                            "name": name,
                            "city": city,
                            "district": district,
                            "message": f"Risk skoru son snapshot'a göre {risk_delta:.1f} puan arttı.",
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
    """
    Kapasite, veteriner, sahiplendirme ve kısırlaştırma senaryosunun
    risk skoruna etkisini hesaplar.
    """

    from src.risk import calculate_risk

    if df.empty:
        return pd.DataFrame()

    scenario_df = df.copy()

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
    scenario_df["sterilization_count"] = (
        scenario_df["sterilization_count"] + extra_sterilizations
    )

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

    scenario_df = scenario_df[output_cols].sort_values(
        "risk_score_improvement",
        ascending=False,
    )

    return scenario_df
