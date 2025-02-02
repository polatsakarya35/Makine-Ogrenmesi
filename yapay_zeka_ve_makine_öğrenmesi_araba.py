# 1. Gerekli kütüphaneleri içe aktar
import numpy as np  # Sayılarla çalışmak için
import pandas as pd  # Verileri düzenlemek için
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine ayırmak için
from sklearn.feature_extraction.text import TfidfVectorizer  # Metinleri sayılara çevirmek için
from sklearn.preprocessing import LabelEncoder  # Etiketleri sayısal hale getirmek için
from tensorflow.keras.models import Sequential  # Derin öğrenme modeli oluşturmak için
from tensorflow.keras.layers import Dense, Dropout  # Katmanlar eklemek için
from tensorflow.keras.optimizers import Adam  # Modeli optimize etmek için

# 2. Örnek veri seti oluştur
# Bu veri, arabaların nasıl sürüleceğini öğreten yol tarifleridir.
# Örneğin "Bu araba çok hızlı" gibi cümleler.
data = {
    "text": [
        "Bu ürün harika, kesinlikle tavsiye ederim!",  # Bu yol hızlı
        "Çok kötü bir deneyim yaşadım, para tuzağı.",  # Bu yol yanlış
        "Hızlı kargo ve kaliteli ürün, teşekkürler.",  # Bu yol doğru
        "Ürün beklentilerimin çok altında, hayal kırıklığı.",  # Bu yol yanlış
        "Fiyatına göre oldukça iyi, memnun kaldım.",  # Bu yol doğru
        "Kargo gecikti ve ürün hasarlı geldi."  # Bu yol yanlış
    ],
    "label": ["olumlu", "olumsuz", "olumlu", "olumsuz", "olumlu", "olumsuz"]  # Yolların ne kadar iyi olduğunu gösteren etiketler
}

# 3. Veriyi DataFrame'e dönüştür
df = pd.DataFrame(data)  # Yol tariflerini ve etiketleri düzenli bir formda saklıyoruz

# 4. Metin verisini ve etiketleri ayır
texts = df["text"].values  # Yollar (metinler)
labels = df["label"].values  # Etiketler (olumlu veya olumsuz)

# 5. Etiketleri sayısallaştır (örneğin, "olumlu" -> 1, "olumsuz" -> 0)
label_encoder = LabelEncoder()  # Etiketlerin sayısal hale getirilmesi
labels = label_encoder.fit_transform(labels)  # "olumlu" = 1, "olumsuz" = 0

# 6. Metin verisini sayısallaştır (TF-IDF vektörleme)
vectorizer = TfidfVectorizer(max_features=1000)  # Tfidf, metin verisini sayılara çeviren bir yöntem
X = vectorizer.fit_transform(texts).toarray()  # Metinleri vektörlere dönüştürüp sayılara çeviriyoruz

# 7. Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
# Burada veriyi, arabaya **eğitim yolları** ve **test yolları** olarak ayırıyoruz
# %80 eğitim, %20 test verisi

# 8. Derin öğrenme modelini oluştur (Yolculuk için araba)
model = Sequential([  # Modelin katmanlarını ekliyoruz
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),  # İlk katman, arabaya yol göstermeye başlıyor
    Dropout(0.5),  # Aşırı öğrenmeyi önlemek için, araba bazen yanlış yola sapabilir
    Dense(64, activation="relu"),  # İkinci katman, arabaya daha iyi sürüş bilgisi veriyor
    Dropout(0.5),  # Yine hataları önlemek için
    Dense(1, activation="sigmoid")  # Son katman, arabaya doğru tahmini yapmayı öğretir
])

# 9. Modeli derle (Araba nasıl çalışacak?)
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
# Adam, arabaya nasıl daha iyi sürmesi gerektiğini öğretir (optimizer)
# binary_crossentropy, arabanın doğru yolu bulması için hataları ölçer

# 10. Modeli eğit (Araba eğitim sırasında yol alıyor)
history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2)
# Modeli 10 defa (epoch) eğitiyoruz, her defasında daha iyi öğrenmesi için
# Batch_size, her seferinde eğitim için kaç örnek kullanacağımızı belirler
# Validation_split, eğitim sırasında %20'yi test için ayırır

# 11. Modeli değerlendir (Test yolunda arabayı deniyoruz)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Kaybı (Loss): {loss}")  # Test yolundaki hataları gösterir
print(f"Test Doğruluğu (Accuracy): {accuracy}")  # Test yolundaki başarıyı gösterir

# 12. Yeni metinler üzerinde tahmin yap (Araba yeni yolları nasıl gider?)
new_texts = [
    "Kesinlikle tavsiye ederim, harika bir ürün!",  # Bu yol doğru
    "Kötü bir deneyim yaşadım, ürün bozuk çıktı."  # Bu yol yanlış
]
new_X = vectorizer.transform(new_texts).toarray()  # Yeni yol tariflerini sayılara çeviriyoruz
predictions = model.predict(new_X)  # Model, yeni yolları test eder
predicted_labels = label_encoder.inverse_transform((predictions > 0.5).astype(int).flatten())  # Yolları olumlu ya da olumsuz olarak etiketler

# 13. Sonuçları yazdır
for text, label in zip(new_texts, predicted_labels):
    print(f"Metin: {text} -> Tahmin: {label}")  # Araba yeni yolları nasıl gittiğini gösterir
