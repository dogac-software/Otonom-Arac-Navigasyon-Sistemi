# SVM Güvenlik Sınırı — Otonom Araç Navigasyon Modülü

## Proje Özeti

Bu proje, iki farklı engel sınıfını **maksimum güvenlik koridoru** ile ayıran
matematiksel modeli (Hard-Margin SVM) Java ile implement etmektedir.

Algoritma, Sequential Minimal Optimization (SMO) yöntemiyle iki sınıf arasındaki
**en geniş marjinli hiper-düzlemi** bulur.

---

## Matematiksel Temel

**Primal optimizasyon problemi:**

```
minimize    ½ ||w||²
subject to  yᵢ(w·xᵢ + b) ≥ 1    ∀ i
```

**Karar sınırı:** `w₀·x + w₁·y + b = 0`

**Marjin genişliği:** `2 / ||w||`

Destek vektörleri (en yakın noktalar) `yᵢ(w·xᵢ + b) = 1` üzerinde yer alır.

---

## Neden Optimal?

- Optimizasyon problemi **konveks** ve **dışbükey** → küresel minimum garantili
- Hiçbir başka ayırıcı çizgi bu marjinden daha geniş olamaz (QP kanıtı)
- Karar sınırı her iki sınıfın en yakın noktasına eşit mesafededir
- Bu nedenle güvenlik koridoru **matematiksel anlamda maksimum**'dur

---

## Zaman Karmaşıklığı (Big-O Analizi)

| Operasyon               | Zaman         | Alan  |
|-------------------------|---------------|-------|
| Başlatma (initState)    | O(n)          | O(n)  |
| Kernel hesabı           | O(1)  (d=2)   | O(1)  |
| Karar fonksiyonu        | O(n)          | O(1)  |
| SMO tekil iterasyon     | O(n²)         | O(1)  |
| **Toplam (ortalama)**   | **O(n²)**     | O(n)  |
| Toplam (en kötü durum)  | O(n³)         | O(n)  |

---

## Yazılım Mimarisi (OOP)

```
svm/
├── model/
│   ├── DataPoint.java       # İmmutable veri noktası (x, y, label)
│   └── SVMResult.java       # İmmutable sonuç (w, b, margin, SVs)
├── data/
│   └── Dataset.java         # Veri seti fabrika metodları
├── algorithm/
│   ├── SVMAlgorithm.java    # SMO algoritması (ana iş mantığı)
│   └── SVMValidator.java    # Doğrulama ve raporlama
├── visualization/
│   └── ConsoleVisualizer.java # ASCII görselleştirme
├── utils/
│   └── ComplexityAnalyzer.java # Big-O analizi ve benchmark
└── Main.java                # Giriş noktası
```

**OOP Prensipleri:**
- **Encapsulation**: Tüm alanlar private, defensive copy kullanımı
- **Immutability**: DataPoint ve SVMResult immutable (bellek güvenliği)
- **Single Responsibility**: Her sınıfın tek sorumluluğu
- **Factory Method**: Dataset fabrika metodları
- **Separation of Concerns**: Model / Data / Algorithm / View katmanları

**Bellek Güvenliği:**
- `DataPoint` ve `SVMResult` immutable → paylaşımda güvenli
- `getWeights()` defensive copy döndürür → dış mutasyona kapalı
- `getPoints()` unmodifiableList döndürür
- Training state (alpha dizisi) fit() sonrası GC tarafından alınır

---

## Kurulum ve Çalıştırma

**Gereksinim:** Java 17+

```bash
# 1. Derleme + ana demo
chmod +x build.sh
./build.sh

# 2. Unit testler
./build.sh test

# 3. Her ikisi
./build.sh all
```

**Manuel derleme:**
```bash
mkdir -p out
find src/main/java -name "*.java" | xargs javac -d out
find src/test/java -name "*.java" | xargs javac -cp out -d out
java -cp out svm.Main
java -cp out svm.SVMTest
```

---

## Senaryolar

1. **Geniş Ayrım Demosu** — İki uzak küme, geniş marjin
2. **Dar Marjin Demosu** — Birbirine yakın kümeler
3. **Rastgele Veri Seti** — Gaussian dağılım, tekrarlanabilir (seed)
4. **Özel Koordinatlar** — Ödevde verilen koordinat setinden örnek

---

## Çıktı Açıklaması

```
+ → Pozitif sınıf (engel tipi A)
- → Negatif sınıf (engel tipi B)
S → Destek vektörü (en yakın noktalar)
| → Karar sınırı çizgisi
: → Marjin çizgileri (güvenlik koridoru kenarları)
```

---

## Referanslar

- Platt, J. (1998). *Sequential Minimal Optimization: A Fast Algorithm for Training SVMs*
- Vapnik, V. (1995). *The Nature of Statistical Learning Theory*
- Cortes & Vapnik (1995). *Support-Vector Networks, Machine Learning*
