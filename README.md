# SmartShelter GIS

## Amaç
Hayvan bakımevleri ve toplama merkezleri için açık veri tabanlı GIS karar destek prototipi.

## Özellikler
- Canlı CKAN API veri denemesi
- Stabil demo CSV fallback
- Harita tabanlı merkez görüntüleme
- Kapasite/doluluk analizi
- Risk skoru
- İlçe ve risk filtreleme
- Veri kalite notları
- CSV/Excel rapor indirme

## Risk Skoru Metodolojisi
Bu skor resmi denetim skoru değildir. Demo amaçlı karar destek göstergesidir.

## Veri Kaynakları
- Demo CSV
- Belediye açık veri portalları
- CKAN uyumlu API kaynakları

## Çalıştırma
pip install -r requirements.txt
streamlit run app.py

## Yol Haritası
- PostGIS entegrasyonu
- İl/ilçe sınır katmanları
- Ulusal veri standardı
- Mobil saha veri girişi
- Bakanlık API entegrasyonu
