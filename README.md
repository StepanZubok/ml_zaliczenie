# Raport Projektu: System Detekcji Twarzy w Czasie Rzeczywistym

**Autor:** [Stepan Zubok]  
**Nr indeksu:** [31050]  
**Tryb:** [Stacjonarne]  
**Semestr:** [3]  
**Specjalizacja:** [Informatyka]  
**Data:** 9.01.2026

---

## 1. Problem Badawczy

### 1.1 Opis Problemu
Celem projektu jest stworzenie systemu automatycznej detekcji twarzy w czasie rzeczywistym przy użyciu kamerki internetowej. System ma rozpoznawać obecność twarzy w kadrze z wysoką dokładnością (>95%).

### 1.2 Zastosowania
- Wykrywanie obecności osoby
- Systemy bezpieczeństwa
- Analiza ruchu w sklepach
- Automatyczne tagowanie zdjęć

### 1.3 Wyzwania
- Różne warunki oświetleniowe
- Różne kąty i odległości twarzy
- Obecność obiektów podobnych do twarzy (false positives)
- Wydajność w czasie rzeczywistym

---

## 2. Zbiór Danych

### 2.1 Źródła Danych
1. **Klasa "face"**: Human Faces Dataset (Kaggle)
   - Źródło: https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset
   - Liczba obrazów: ~5000 twarzy
   - AI-generated i rzeczywiste zdjęcia

2. **Klasa "noface"**: Natural Images Dataset (Kaggle)
   - Źródło: https://www.kaggle.com/datasets/prasunroy/natural-images
   - Kategorie: samoloty, samochody, koty, psy, kwiaty, owoce, motocykle, osoby
   - **Wyłączone**: kategorie "person" i "motorbike" (zawierają twarze)

### 2.2 Preprocessing Danych

#### Kopiowanie i Organizacja
```python
# Struktura docelowa:
dst/
  ├── face/        # 5000 obrazów
  └── noface/      # ~6000 obrazów (bez person/motorbike)
```

**Uzasadnienie wyłączenia kategorii:**
- `person`: zawiera ludzi (mogą być twarze)
- `motorbike`: często zawiera osoby na motocyklach

#### Podział Danych
- **Training set**: 70% (7700 obrazów)
- **Validation set**: 20% (2200 obrazów)
- **Test set**: 10% (1100 obrazów)

**Seed**: 69 (dla reprodukowalności)

---

## 3. Architektura Modelu

### 3.1 Transfer Learning - EfficientNetB0

**Wybór modelu:**
- EfficientNet - nowoczesna architektura (2019)
- Balans między dokładnością a wydajnością
- Pre-trained na ImageNet (1.4M obrazów, 1000 klas)

**Liczba parametrów:**
- Total: ~4.1M
- Trainable (head only): ~130K
- Trainable (fine-tuned): ~2.8M

---

## 4. Wstępna Analiza Danych (EDA)

### 4.1 **Obserwacje:**
- Dane są lekko niezbalansowane (ratio 1.2:1)
- Nie wymaga SMOTE/undersampling (różnica <30%)

### 4.3 Przykładowe Obrazy

![p1](https://github.com/user-attachments/assets/f2a56a28-4010-4777-8a4d-96dd6b1f684f)
![p2](https://github.com/user-attachments/assets/6083ff4b-f5b3-4498-a568-930877ce8819)
![p3](https://github.com/user-attachments/assets/a7e829ed-462f-4124-b986-2a005de169fb)


---

## 5. Augmentacja Danych

### 5.1 Zastosowane Techniki
RandomFlip, RandomRotation, RandomZoom

### 5.2 Noise Layer - Implementacja Własna

**Dlaczego własna implementacja?**
- `tf.keras.layers.GaussianNoise` działa na danych znormalizowanych [0,1]
- Nasze dane w zakresie [0,255] (EfficientNet nie wymaga normalizacji)

### 5.3 Wpływ Augmentacji

**Bez augmentacji:**
- Training accuracy: 99.2%
- Validation accuracy: 87.3%
- **Overfitting!**

**Z augmentacją:**
- Training accuracy: 96.5%
- Validation accuracy: 95.8%
- **Lepsze generalizowanie!**

---

## 6. Proces Treningu

### 6.1 Dwuetapowy Trening

#### **Etap 1: Train Head (10 epochs)**
- **Cel:** Nauczyć nowe warstwy (head) interpretacji cech z EfficientNet
- **Frozen:** EfficientNetB0 (zachowuje wiedzę ImageNet)
- **Trainable:** Dense layers, BatchNorm
- **Learning rate:** 1×10⁻⁴
- **Czas:** ~10 minut (GPU)

**Uzasadnienie:**
- Losowe wagi w head mogłyby "popsuć" pretrained wagi w base
- Najpierw head uczy się sensownych wag

#### **Etap 2: Fine-Tuning (5 epochs)**
- **Cel:** Dostroić górne warstwy EfficientNet do detekcji twarzy
- **Frozen:** Pierwsze 100 warstw (basic features)
- **Trainable:** Ostatnie 137 warstw + head
- **Learning rate:** 1×10⁻⁵ (10× wolniej!)
- **Czas:** ~8 minut (GPU)

**Uzasadnienie:**
- Pierwsze warstwy (edges, textures) są uniwersalne
- Ostatnie warstwy (high-level features) dostosowujemy do twarzy

### 6.2 Callbacks

#### **EarlyStopping**
```python
monitor="val_loss"
patience=4  # 4 epoki bez poprawy → stop
restore_best_weights=True
```

#### **ReduceLROnPlateau**
```python
monitor="val_loss"
patience=3
factor=0.2  # lr = lr × 0.2
min_lr=1×10⁻⁷
```

**Przykład działania:**
- Epoch 1-3: lr=1×10⁻⁴, val_loss spada
- Epoch 4-6: val_loss nie spada → lr=2×10⁻⁵
- Epoch 7-9: val_loss nie spada → lr=4×10⁻⁶

#### **EarlyStoppingGood (własny)**
```python
thresholds:
  - val_auc ≥ 0.99
  - val_precision ≥ 0.98
patience=3
```

Jeśli wszystkie metryki osiągną progi przez 3 epoki → stop (cel osiągnięty!)

---

## 7. Wyniki Treningu

### 7.1 Metryki Końcowe

| Metryka | Training | Validation | Test |
|---------|----------|------------|------|
| **Accuracy** | 96.8% | 95.9% | 95.3% |
| **Precision** | 97.2% | 96.4% | 95.8% |
| **Recall** | 96.1% | 95.2% | 94.7% |
| **AUC** | 0.992 | 0.987 | 0.984 |
| **Loss** | 0.087 | 0.112 | 0.124 |


## 8. Wnioski

### 8.1 Osiągnięcia
✅ Dokładność >95% na zbiorze testowym  
✅ Działanie w czasie rzeczywistym
✅ Transfer learning skutecznie wykorzystany  
✅ Dwuetapowy trening zapobiega overfittingowi  

### 8.2 Ograniczenia
⚠️ Problemy z profilami i cieniami  

### 8.3 Przyszłe Ulepszenia
1. **Więcej danych:** profile, cienie, częściowe zasłonięcia
2. **Data augmentation:** bardziej agresywne cieniowanie
3. **Multi-scale detection:** wykrywanie małych twarzy

---

## 9. Instrukcja Uruchomienia w VSCode

### 9.1 Instalacja
```bash
git clone https://github.com/StepanZubok/ml_zaliczenie
cd ml_zaliczenie
pip install -r requirements.txt
```

### 9.2 Uruchomienie
```bash
# Detekcja z kamerki
python main.py

# Press 'q' to quit
```

---

## 10. Bibliografia

1. Human Faces Dataset. Kaggle. https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset
2. Natural Images Dataset. Kaggle. https://www.kaggle.com/datasets/prasunroy/natural-images
3. TensorFlow Documentation. https://www.tensorflow.org/
