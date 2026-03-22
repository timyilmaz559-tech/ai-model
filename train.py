import tensorflow as tf
import cv2
import numpy as np
import os
import sys

# 1. Veriyi yükle
X, y = [], []
siniflar = ['CLASS_NAME1', 'CLASS_NAME2']#you can add more class

# Klasör kontrolü ekle
for sinif_id, sinif_adi in enumerate(siniflar):
    klasor_yolu = f'dataset/{sinif_adi}'
    
    # Klasör yoksa hata verme, geç
    if not os.path.exists(klasor_yolu):
        print(f"⚠️ Uyarı: {klasor_yolu} klasörü bulunamadı!")
        continue
        
    for dosya in os.listdir(klasor_yolu):
        if not dosya.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Sadece resim dosyalarını al
            
        img_path = f'dataset/{sinif_adi}/{dosya}'
        img = cv2.imread(img_path)
        
        if img is None:  # Boş dosya kontrolü
            print(f"⚠️ {dosya} yüklenemedi, atlanıyor...")
            continue
            
        img = cv2.resize(img, (64, 64))
        X.append(img)
        y.append(sinif_id)

# Veri yüklenmediyse hata ver
if len(X) == 0:
    print("❌ Hiç görüntü yüklenemedi! Klasör yapısını kontrol et:")
    print("   ├── dataset/")
    print("   │   ├── Muaz/")
    print("   │   └── other/")
    sys.exit(1)

X = np.array(X) / 255.0
y = np.array(y)

print(f"✅ Yüklenen görüntü sayısı: {len(X)}")
print(f"📊 Sınıf dağılımı: Muaz={sum(y==0)}, other={sum(y==1)}")

# 2. CNN modeli oluştur (4 sınıf değil, 2 sınıf olmalı!)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(siniflar), activation='softmax')  # Dinamik sınıf sayısı
])

# 3. Modeli derle ve eğit
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# validation_split kullanırken verinin karıştırıldığından emin ol
model.fit(X, y, epochs=50, validation_split=0.2, shuffle=True)

# 4. Modeli kaydet
model.save('face.h5')
print("✅ Model kaydedildi: face.h5")
