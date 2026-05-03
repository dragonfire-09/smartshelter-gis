import pandas as pd


# ---------------------------------------------------------
# Risk Calculation
# ---------------------------------------------------------
def calculate_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Risk skoru hesaplar. Yalnızca risk_eligible kayıtlar için.
    Eksik veriye dayanarak fabrikasyon yapmaz.
    """
    df = df.copy()

    df["risk_score"] = pd.NA
    df["risk_level"] = "Veri yetersiz"

    if "risk_eligible" not in df.columns:
        df["risk_eligible"] = False
        return df

    eligible_mask = df["risk_eligible"].fillna(False).astype(bool)

    if not eligible_mask.any():
        return df

    sub = df[eligible_mask].copy()

    occ_rate = pd.to_numeric(sub.get("occupancy_rate"), errors="coerce").fillna(0)
    animals_per_vet = pd.to_numeric(sub.get("animals_per_vet"), errors="coerce").fillna(0)
    capacity = pd.to_numeric(sub.get("capacity"), errors="coerce").fillna(0)
    occupancy = pd.to_numeric(sub.get("occupancy"), errors="coerce").fillna(0)
    sterilization = pd.to_numeric(sub.get("sterilization_count"), errors="coerce").fillna(0)
    adoption = pd.to_numeric(sub.get("adoption_count"), errors="coerce").fillna(0)

    # Doluluk baskısı: 0-100 doluluk %.
    occupancy_pressure = occ_rate.clip(0, 150)

    # Veteriner baskısı: 0-200 hayvan/veteriner -> 0-100 puan.
    vet_pressure = (animals_per_vet.clip(0, 200) / 200) * 100

    # Sahiplendirme/Kısırlaştırma performansı
    # Yüksek mevcut hayvana karşı düşük sahiplendirme + kısırlaştırma => risk artar.
    operational_baseline = (sterilization + adoption).clip(lower=0)

    operational_ratio = pd.Series(0.0, index=sub.index)
    safe_occ_mask = occupancy > 0
    operational_ratio.loc[safe_occ_mask] = (
        operational_baseline.loc[safe_occ_mask] / occupancy.loc[safe_occ_mask]
    ).clip(0, 1)

    # 1.0 ratio => 0 risk katkısı, 0.0 ratio => 100 risk katkısı
    operational_pressure = (1 - operational_ratio) * 100

    # Ağırlıklı toplam
    score = (
        occupancy_pressure * 0.45
        + vet_pressure * 0.30
        + operational_pressure * 0.25
    )

    score = score.clip(0, 100).round(1)

    df.loc[eligible_mask, "risk_score"] = score

    df.loc[eligible_mask & (score < 40), "risk_level"] = "Düşük"
    df.loc[
        eligible_mask & (score >= 40) & (score < 70),
        "risk_level",
    ] = "Orta"
    df.loc[eligible_mask & (score >= 70), "risk_level"] = "Kritik"

    return df


# ---------------------------------------------------------
# Action Recommendations
# ---------------------------------------------------------
def create_action_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "recommended_action" not in df.columns:
        df["recommended_action"] = ""

    risk_eligible = (
        df["risk_eligible"].fillna(False).astype(bool)
        if "risk_eligible" in df.columns
        else pd.Series([False] * len(df), index=df.index)
    )

    has_coord = (
        df["coordinate_valid"].fillna(False).astype(bool)
        if "coordinate_valid" in df.columns
        else pd.Series([False] * len(df), index=df.index)
    )

    has_capacity = (
        df["capacity_available"].fillna(False).astype(bool)
        if "capacity_available" in df.columns
        else pd.Series([False] * len(df), index=df.index)
    )

    has_occupancy = (
        df["occupancy_available"].fillna(False).astype(bool)
        if "occupancy_available" in df.columns
        else pd.Series([False] * len(df), index=df.index)
    )

    occ_rate = pd.to_numeric(df.get("occupancy_rate"), errors="coerce")
    animals_per_vet = pd.to_numeric(df.get("animals_per_vet"), errors="coerce")

    actions = []

    for idx in df.index:
        if not risk_eligible.loc[idx]:
            problems = []

            if not has_capacity.loc[idx]:
                problems.append("kapasite verisi")
            if not has_occupancy.loc[idx]:
                problems.append("mevcut hayvan verisi")
            if not has_coord.loc[idx]:
                problems.append("koordinat")

            if problems:
                actions.append(
                    f"Risk hesaplanamadı. Eksik alanlar: {', '.join(problems)}. "
                    "Kaynak verinin tamamlanması gerekir."
                )
            else:
                actions.append(
                    "Risk için yeterli veri yok. Kaynak veri kalitesi artırılmalıdır."
                )
            continue

        risk_level = str(df.loc[idx, "risk_level"])
        rate = occ_rate.loc[idx] if pd.notna(occ_rate.loc[idx]) else 0
        apv = animals_per_vet.loc[idx] if pd.notna(animals_per_vet.loc[idx]) else 0

        steps = []

        if rate >= 100:
            steps.append("Kapasite üstü doluluk; acil sahiplendirme/transfer planlanmalı")
        elif rate >= 80:
            steps.append("Yüksek doluluk; sahiplendirme kampanyası ve kısırlaştırma artırılmalı")

        if apv >= 100:
            steps.append("Veteriner başına çok yüksek hayvan; veteriner takviyesi gerekli")
        elif apv >= 50:
            steps.append("Veteriner iş yükü yüksek; ek veteriner desteği değerlendirilmeli")

        if risk_level == "Kritik" and not steps:
            steps.append("Kritik risk; kapsamlı operasyonel inceleme yapılmalı")

        if risk_level == "Düşük" and not steps:
            steps.append("Mevcut operasyonel düzey korunmalı, izleme sürdürülmeli")

        if not steps:
            steps.append("Operasyonel takip ve veri güncellemesi sürdürülmeli")

        actions.append(". ".join(steps) + ".")

    df["recommended_action"] = actions

    return df
