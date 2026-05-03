import pandas as pd


def calculate_risk(df):
    df = df.copy()

    if df.empty:
        df["occupancy_rate"] = []
        df["animals_per_vet"] = []
        df["risk_score"] = []
        df["risk_level"] = []
        return df

    capacity_safe = df["capacity"].replace(0, 1)
    occupancy_safe = df["occupancy"].replace(0, 1)

    df["occupancy_rate"] = (df["occupancy"] / capacity_safe * 100).round(1)

    df["animals_per_vet"] = (
        df["occupancy"] / df["vet_count"].replace(0, 1)
    ).round(1)

    # 1. Kapasite baskısı
    # %150 ve üzeri doluluk maksimum baskı olarak değerlendirilir.
    capacity_pressure = (
        (df["occupancy"] / capacity_safe).clip(0, 1.5) / 1.5 * 100
    )

    # 2. Veteriner iş yükü
    # Veteriner başına 150 hayvan ve üzeri yüksek baskı kabul edilir.
    vet_load = df["occupancy"] / (df["vet_count"] + 1)
    vet_pressure = (vet_load / 150).clip(0, 1) * 100

    # 3. Sahiplendirme açığı
    adoption_gap = (
        1 - (df["adoption_count"] / occupancy_safe)
    ).clip(0, 1) * 100

    # 4. Kısırlaştırma açığı
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
        + adoption_gap * 0.15
        + sterilization_gap * 0.10
    ).clip(0, 100).round(1)

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, 40, 70, 100],
        labels=["Düşük", "Orta", "Kritik"],
    )

    return df


def create_action_recommendations(df):
    df = df.copy()

    if df.empty:
        df["recommended_action"] = []
        return df

    recommendations = []

    for _, row in df.iterrows():
        actions = []

        if row["occupancy_rate"] >= 100:
            actions.append("kapasite artırımı veya sevk planlaması")
        elif row["occupancy_rate"] >= 85:
            actions.append("kapasite yakından izlenmeli")

        if row["animals_per_vet"] >= 150:
            actions.append("veteriner/personel desteği")
        elif row["animals_per_vet"] >= 100:
            actions.append("veteriner iş yükü izlenmeli")

        if row["adoption_count"] < row["occupancy"] * 0.15:
            actions.append("sahiplendirme kampanyası")

        if row["sterilization_count"] < row["occupancy"] * 0.25:
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
