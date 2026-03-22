import speech_recognition as sr
import requests
import pyttsx3

uyandirma = "hey home"
kapatma = "turn off"
uyku = True

url = "http://192.168.0.6:5050/api/generate"

def ai_konus(command):
    global url
    payload = {
        "model": "qwen2.5:1.5b",
        "prompt": command,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            answer = response.json().get("response", "")
            return answer
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Connection error{e}"

def ses_ver(metin):
    tts = pyttsx3.init()
    tts.say(metin)
    tts.runAndWait()

# Mikrofonu başlat
r = sr.Recognizer()
mikrofon = sr.Microphone()

print("AI Asistan Başladı. 'Hey home' deyin...")

while True:
    with mikrofon as kaynak:
        r.adjust_for_ambient_noise(kaynak)
        print("Dinliyorum...")
        ses = r.listen(kaynak, timeout=None)
        
    try:
        metin = r.recognize_google(ses, language="tr-TR").lower()
        print(f"Duyulan: {metin}")
    except sr.WaitTimeoutError:
        continue  # Süre doldu, tekrar dene
    except sr.UnknownValueError:
        continue  # Anlaşılmadı, devam et
    except KeyboardInterrupt:
        print("\nKapatılıyor...")
        break

    if uyandirma in metin and uyku:
        uyku = False
        print("Emir bekliyorum...")
        
        # Komutu al
        with mikrofon as kaynak2:
            komut_ses = r.listen(kaynak2, timeout=5)
            komut = r.recognize_google(komut_ses, language="en-US")
            print(f"Komut: {komut}")
            
            cevap = ai_konus(komut)
            print(f"Cevap: {cevap}")
            ses_ver(cevap)
            uyku = True