from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf

# YOLO11n kullan
detector = YOLO('yolo26n.pt')

# CNN ile yüz tanıma
classifier = tf.keras.models.load_model('face.h5')
siniflar = ['CLASS_NAM1', 'CLASS_NAME2']

# YOLO'da insan sınıfı ID'si 0'dır
INSAN_SINIF_ID = 0

# Filtre parametreleri
MIN_ALAN = 2000       
MAX_ASPECT = 1.5      
MIN_ASPECT = 0.7      
MIN_GUVEN = 0.3       # Geçici olarak düşürdüm

cap = cv2.VideoCapture(0)

# İstatistikler
toplam_insan = 0
taninan_muaz = 0
taninan_diger = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # YOLO ile nesne tespiti
    results = detector(frame)[0]
    
    for box in results.boxes:
        sinif_id = int(box.cls[0])
        yolo_guven = float(box.conf[0])
        
        # Tespit edilen her şeyi göster (teşhis için)
        if sinif_id == INSAN_SINIF_ID:
            etiket = f"Insan (%{yolo_guven*100:.1f})"
            renk = (255, 255, 0)  # Sarı
        else:
            # İnsan değilse de göster (mavi)
            etiket = f"Sınıf {sinif_id} (%{yolo_guven*100:.1f})"
            renk = (255, 0, 0)
            cv2.rectangle(frame, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), 
                         (int(box.xyxy[0][2]), int(box.xyxy[0][3])), renk, 1)
            cv2.putText(frame, etiket, 
                       (int(box.xyxy[0][0]), int(box.xyxy[0][1])-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, renk, 1)
            continue
        
        # Sadece insanları işle
        toplam_insan += 1
        
        # Koordinatlar
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Boyut kontrolü
        w, h = x2-x1, y2-y1
        alan = w * h
        aspect = w / h if h > 0 else 0
        
        # Boyut bilgilerini göster
        cv2.putText(frame, f"Alan:{alan} Aspect:{aspect:.2f}", 
                   (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Filtre geçti mi?
        filtre_gec = True
        if alan < MIN_ALAN:
            cv2.putText(frame, "ALAN KUCUK", (x1, y2+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            filtre_gec = False
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            cv2.putText(frame, "ASPECT HATALI", (x1, y2+45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            filtre_gec = False
        
        if not filtre_gec:
            # Filtreden geçmediyse gri kutu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128,128,128), 1)
            continue
        
        # Yüz bölgesini kırp
        yuz_bolgesi = frame[y1:y2, x1:x2]
        if yuz_bolgesi.size == 0: 
            continue
        
        # CNN ile sınıflandır
        img = cv2.resize(yuz_bolgesi, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB'ye çevir
        img_norm = np.expand_dims(img_rgb, axis=0) / 255.0
        
        tahmin = classifier.predict(img_norm, verbose=0)[0]
        sinif = np.argmax(tahmin)
        guven = tahmin[sinif]
        
        # Tahmin bilgisi
        cv2.putText(frame, f"CNN: Sınıf{sinif} %{guven*100:.1f}", 
                   (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        
        # Güven eşiğini kontrol et
        if guven < MIN_GUVEN:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128,128,128), 2)
            cv2.putText(frame, f"AZ GUVEN (%{guven*100:.1f})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 2)
            continue
        
        # İstatistik
        if sinif == 0:
            taninan_muaz += 1
            renk = (0, 255, 0)
            etiket = f"MUAZ! %{guven*100:.1f}"
        else:
            taninan_diger += 1
            renk = (0, 0, 255)
            etiket = f"DIGER %{guven*100:.1f}"
        
        # Sonucu göster
        cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)
        cv2.putText(frame, etiket, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 2)
    
    # İstatistikleri göster
    cv2.putText(frame, f"Insan: {toplam_insan} | Muaz: {taninan_muaz} | Diger: {taninan_diger}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    cv2.imshow('Yuz Tanima - Teshis Modu', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
