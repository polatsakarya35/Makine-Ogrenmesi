# 1. Gerekli kütüphaneleri içe aktar
import numpy as np  # Sayılarla çalışmak için kullanılan bir kütüphane
import pandas as pd  # Veriyi düzenlemek ve işlem yapmak için kullanılan bir kütüphane
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine ayırmak için
from sklearn.feature_extraction.text import TfidfVectorizer  # Metinleri sayılara dönüştürmek için kullanılan vektörleştirici
from sklearn.preprocessing import LabelEncoder  # Etiketleri sayısal verilere dönüştürmek için
from tensorflow.keras.models import Sequential  # Derin öğrenme modelini oluşturmak için
from tensorflow.keras.layers import Dense, Dropout  # Derin öğrenme modeline katmanlar eklemek için
from tensorflow.keras.optimizers import Adam  # Modeli optimize etmek için kullanılan optimizasyon algoritması

# **Açıklama**:
# - NumPy: Sayısal verilerle çalışmak için kullanılır.
# - Pandas: Verileri düzenler ve analiz eder, tablolarla çalışmayı kolaylaştırır.
# - Scikit-learn: Makine öğrenimi algoritmalarını içerir. Bu kodda metin verisini vektörleştirmek ve veriyi eğitim ve test setlerine ayırmak için kullanılır.
# - TensorFlow/Keras: Derin öğrenme modelleri geliştirmek için kullanılır. Modeli oluşturmak ve eğitmek için kullanılır.

# 2. Örnek veri seti oluştur
data = {
    "text": [
        "Bu ürün harika, kesinlikle tavsiye ederim!",
        "Çok kötü bir deneyim yaşadım, para tuzağı.",
        "Hızlı kargo ve kaliteli ürün, teşekkürler.",
        "Ürün beklentilerimin çok altında, hayal kırıklığı.",
        "Fiyatına göre oldukça iyi, memnun kaldım.",
        "Kargo gecikti ve ürün hasarlı geldi."
    ],
    "label": ["olumlu", "olumsuz", "olumlu", "olumsuz", "olumlu", "olumsuz"]
}

# **Açıklama**:
# - Bu veri seti, olumlu ve olumsuz yorumları içerir.
# - "text" anahtarı altındaki veriler metin yorumları, "label" anahtarı altındaki veriler ise bu yorumların etiketleridir (olumlu/olumsuz).

# 3. Veriyi DataFrame'e dönüştür
df = pd.DataFrame(data)  # DataFrame, verileri düzenli bir tablo (tablo formatında) şeklinde tutmamızı sağlar.

# **Açıklama**:
# - Pandas DataFrame, verileri düzenli bir şekilde tutmaya ve işlemenize olanak tanır. Yani veriler üzerinde kolayca manipülasyon yapabiliriz.

# 4. Metin verisini ve etiketleri ayır
texts = df["text"].values  # Metin verisi: Yorumları alıyoruz.
labels = df["label"].values  # Etiket verisi: Her yorumun etiketini alıyoruz (olumlu veya olumsuz).

# **Açıklama**:
# - Burada veri setimizden **text** (yorumlar) ve **label** (etiketler) verilerini ayırıyoruz. Yorumlar, metin olarak işlenecek ve etiketler, makine öğrenmesi için sayısal verilere dönüştürülecek.

# 5. Etiketleri sayısallaştır (örneğin, "olumlu" -> 1, "olumsuz" -> 0)
label_encoder = LabelEncoder()  # Etiketleri sayısal hale getiren bir fonksiyon
labels = label_encoder.fit_transform(labels)  # "olumlu" = 1, "olumsuz" = 0

# **Açıklama**:
# - **LabelEncoder** sınıfı, etiketleri sayılara dönüştürür. Bu, bilgisayarın **"olumlu"** veya **"olumsuz"** gibi kelimeleri anlamasını sağlar.
#   Örneğin, "olumlu" etiketini 1'e, "olumsuz" etiketini ise 0'a dönüştürür.

# 6. Metin verisini sayısallaştır (TF-IDF vektörleme)
vectorizer = TfidfVectorizer(max_features=1000)  # Metinleri sayısal verilere dönüştürmek için kullanılan araç
X = vectorizer.fit_transform(texts).toarray()  # Metinleri vektörler haline getiriyoruz

# **Açıklama**:
# - **TfidfVectorizer**, metin verilerini sayısal verilere dönüştüren bir araçtır. Metin içindeki kelimelerin önemini analiz eder.
# - Bu işlem sonucunda her bir metin, sayı dizileri (vektörler) olarak temsil edilir. Bu sayılar, metindeki kelimelerin **önemini** gösterir.

# 7. Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
# Verinin %80'ini eğitim için, %20'sini ise test için ayırıyoruz

# **Açıklama**:
# - **train_test_split**, veriyi **eğitim ve test** setlerine ayırmamıza yarar. Eğitim verisi, modelin öğrenmesini sağlarken, test verisi modelin ne kadar doğru tahmin yapabildiğini ölçer.

# 8. Derin öğrenme modelini oluştur
model = Sequential([  # Derin öğrenme modeli başlatılıyor
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),  # İlk katman: Girdi verisiyle çalışacak
    Dropout(0.5),  # Dropout: Aşırı öğrenmeyi engellemek için
    Dense(64, activation="relu"),  # İkinci katman: Modeli daha fazla öğrenmeye yönlendiriyor
    Dropout(0.5),  # Dropout tekrar ekleniyor
    Dense(1, activation="sigmoid")  # Son katman: Çıktı verisi, 1 veya 0 (olumlu ya da olumsuz)
])

# **Açıklama**:
# - **Sequential**: Keras'ta kullanılan bir model yapısıdır. Burada, ardışık (sıralı) katmanlardan oluşan bir model oluşturuluyor.
# - **Dense** katmanları, her katmanda birden fazla nöron (düğüm) bulundurur. Bu nöronlar, verinin işlenmesine yardımcı olur.
# - **ReLU (Rectified Linear Unit)**, nöronların aktivasyon fonksiyonudur. Bu fonksiyon, negatif değerleri sıfır yaparak pozitif değerleri olduğu gibi bırakır.
# - **Dropout**: Bu katman, modelin aşırı öğrenmesini (overfitting) engellemeye yardımcı olur, yani modelin eğitim verisine çok fazla adapte olmasını sağlar.
# - **Sigmoid**: Son katmanda, **ikili sınıflandırma** için kullanılır. Çıktı 1 veya 0 olabilir. "1", olumlu, "0" ise olumsuz anlamına gelir.

# 9. Modeli derle (modelin çalışma prensibini belirlemek)
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
# Adam optimizasyon algoritması kullanarak modelin eğitimini sağlıyoruz

# **Açıklama**:
# - **compile**: Modelin çalışma prensibini belirleriz. Burada:
#   - **Adam**: Bir optimizasyon algoritmasıdır. Modelin doğru tahmin yapabilmesi için hata payını minimize eder.
#   - **binary_crossentropy**: Bu, **ikili sınıflandırma** için kullanılan kayıp fonksiyonudur. Modelin tahmin ettiği etiket ile gerçek etiket arasındaki farkı ölçer.
#   - **accuracy**: Modelin doğruluğunu ölçmek için kullanılan bir metrik.

# 10. Modeli eğit (Model, eğitim verisi ile öğrenir)
history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2)
# Model, 10 defa (epoch) eğitim alır, her defasında öğrenmeye devam eder

# **Açıklama**:
# - **fit** fonksiyonu, modelin eğitimini başlatır. Eğitim verisi (X_train ve y_train) ile model, nasıl çalışması gerektiğini öğrenir.
# - **epochs**: Eğitim veri seti üzerinde ne kadar zaman çalışacağını belirtir.
# - **batch_size**: Her bir eğitim adımında modelin çalışacağı veri miktarını belirtir.
# - **validation_split**: Verinin %20'sini doğrulama için ayırır, böylece eğitim sırasında modelin performansı gözlemlenir.

# 11. Modeli değerlendir (Modelin doğruluğunu test etmek)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Kaybı (Loss): {loss}")
print(f"Test Doğruluğu (Accuracy): {accuracy}")

# **Açıklama**:
# - **evaluate** fonksiyonu, modelin test verisi ile ne kadar başarılı olduğunu ölçer.
#   - **Loss**: Modelin yaptığı hatayı gösterir, yani ne kadar yanlış tahmin yaptığını.
#   - **Accuracy**: Modelin doğruluğunu gösterir, yani doğru tahminlerin oranı.

# 12. Yeni verilerle tahmin yap (Yeni yorumları sınıflandır)
new_texts = [
    "Kesinlikle tavsiye ederim, harika bir ürün!",  # Bu yol doğru
    "Kötü bir deneyim yaşadım, ürün bozuk çıktı."  # Bu yol yanlış
]
new_X = vectorizer.transform(new_texts).toarray()  # Yeni yorumları sayısal verilere dönüştürürüz
predictions = model.predict(new_X)  # Model ile tahmin yaparız

# **Açıklama**:
# - **transform** fonksiyonu, yeni yorumları modelin anlayabileceği sayılara dönüştürür.
# - **predict** fonksiyonu, modelin bu yeni yorumlara verdiği tahminleri döndürür. Bu tahminler 0 veya 1 olacaktır (0 olumsuz, 1 olumlu).
