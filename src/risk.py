import pandas as pd


def calculate_risk(df):
    df = df.copy()

    if df.empty:
        df["occupancy_rate"] = []
        df["animals_per_vet"] = []
        df["risk_score"] = []
        df["risk_level"] = []
        return df

    if "risk_eligible" not in df.columns:
        df["risk_eligible"] = True

    capacity_safe = df["capacity"].replace(0, pd.NA)
    occupancy_safe = df["occupancy"].replace(0, pd.NA)

    df["occupancy_rate"] = (
        df["occupancy"] / capacity_safe * 100
    ).round(1)

    df["animals_per_vet"] = (
        df["occupancy"] / df["vet_count"].replace(0, 1)
    ).round(1)

    capacity_pressure = (
        (df["occupancy"] / capacity_safe).clip(0, 1.5) / 1.5 * 100
    )

    vet_load = df["occupancy"] / (df["vet_count"] + 1)
    vet_pressure = (vet_load / 150).clip(0, 1) * 100

    adoption_gap = (
        1 - (df["adoption_count"] / occupancy_safe)
    ).clip(0, 1) * 100

    sterilization_gap = (
        1 - (df["sterilization_count"] / occupancy_safe)
    ).clip(0, 1) * 100

    df["capacity_pressure_score"] = capacity_pressure.round(1)
    df["vet_pressure_score"] = vet_pressure.round(1)
    df["adoption_gap_score"] = adoption_gap.round(1)
    df["sterilization_gap_score"] = sterilization_gap.round(1)

    df["risk_score"] = (
        capacity_pressure * 0.50
        + vet_pressure * 0.25
        + adoption_gap.fillna(100) * 0.15
        + sterilization_gap.fillna(100) * 0.10
    ).clip(0, 100).round(1)

    # Risk için yeterli veri yoksa risk üretme
    not_ready = ~df["risk_eligible"].astype(bool)

    df.loc[not_ready, "risk_score"] = pd.NA
    df.loc[not_ready, "occupancy_rate"] = pd.NA
    df.loc[not_ready, "animals_per_vet"] = pd.NA

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, 40, 70, 100],
        labels=["Düşük", "Orta", "Kritik"],
    )

    df["risk_level"] = df["risk_level"].astype("object")
    df.loc[not_ready, "risk_level"] = "Veri Yetersiz"

    return df


def create_action_recommendations(df):
    df = df.copy()

    if df.empty:
        df["recommended_action"] = []
        return df

    recommendations = []

    for _, row in df.iterrows():
        actions = []

        if not bool(row.get("risk_eligible", True)):
            if row.get("data_scope") == "capacity_only":
                actions.append("mevcut hayvan sayısı belediyeden doğrulanmalı")
                actions.append("kapasite envanteri risk analizi için doluluk verisiyle tamamlanmalı")
            elif row.get("data_scope") == "location_only":
                actions.append("kapasite ve mevcut hayvan bilgisi eklenmeli")
            else:
                actions.append("kaynak veri ana risk analizi için uygun değil")
            recommendations.append(", ".join(actions).capitalize() + ".")
            continue

        occupancy_rate = row.get("occupancy_rate", 0)
        animals_per_vet = row.get("animals_per_vet", 0)

        if occupancy_rate >= 100:
            actions.append("kapasite artırımı veya sevk planlaması")
        elif occupancy_rate >= 85:
            actions.append("kapasite yakından izlenmeli")

        if animals_per_vet >= 150:
            actions.append("veteriner/personel desteği")
        elif animals_per_vet >= 100:
            actions.append("veteriner iş yükü izlenmeli")

        if row.get("adoption_count", 0) < row.get("occupancy", 0) * 0.15:
            actions.append("sahiplendirme kampanyası")

        if row.get("sterilization_count", 0) < row.get("occupancy", 0) * 0.25:
            actions.append("kısırlaştırma operasyonu")

        if not row.get("coordinate_valid", True):
            actions.append("koordinat/veri güncellemesi")

        if row.get("is_estimated", False):
            actions.append("kaynak veri doğrulaması")

        if not actions:
            actions.append("rutin izleme")

        recommendations.append(", ".join(actions).capitalize() + ".")

    df["recommended_action"] = recommendations

    return df
