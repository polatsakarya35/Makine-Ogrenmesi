# Gerekli kütüphaneleri içe aktar
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Örnek veri seti oluştur (gerçek projelerde CSV veya API'den veri çekebilirsiniz)
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

# Veriyi DataFrame'e dönüştür
df = pd.DataFrame(data)

# Metin verisini ve etiketleri ayır
texts = df["text"].values
labels = df["label"].values

# Etiketleri sayısallaştır (örneğin, "olumlu" -> 1, "olumsuz" -> 0)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Metin verisini sayısallaştır (TF-IDF vektörleme)
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts).toarray()

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Derin öğrenme modeli oluştur
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.5),  # Aşırı öğrenmeyi önlemek için
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # İkili sınıflandırma için sigmoid
])

# Modeli derle
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Modeli eğit
history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2)

# Modeli değerlendir
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Kaybı (Loss): {loss}")
print(f"Test Doğruluğu (Accuracy): {accuracy}")

# Yeni metinler üzerinde tahmin yap
new_texts = [
    "Kesinlikle tavsiye ederim, harika bir ürün!",
    "Kötü bir deneyim yaşadım, ürün bozuk çıktı."
]
new_X = vectorizer.transform(new_texts).toarray()
predictions = model.predict(new_X)
predicted_labels = label_encoder.inverse_transform((predictions > 0.5).astype(int).flatten())

for text, label in zip(new_texts, predicted_labels):
    print(f"Metin: {text} -> Tahmin: {label}")