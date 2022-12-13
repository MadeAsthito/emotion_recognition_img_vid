import cv2
import sys
import numpy as np
from time import sleep
from keras.models import load_model
from keras.utils import img_to_array
from keras.preprocessing import image

# Insialisasi lokasi cascade file dan model file
cascPath = "haarcascade_frontalface_default.xml"
modelPath = "./model/emotion_face_mobilNet.h5"

# Inisialisasi face cascade
faceCascade = cv2.CascadeClassifier(cascPath)
# Inisialisasi Model
model = load_model(modelPath)

# Inisialisasi Label
# ex: class_labels = ('Angry','Happy','Neutral','Sad','Surprise')
class_labels = ('Angry','Disgust','Fear', 'Happy','Neutral', 'Sad','Surprise')

# Capture video
cap = cv2.VideoCapture(0) # 0 --> Internal Webcam

# Selama video ter-capture atau menyala
while True:
    # Mengambil input dari kamera
    ret, frame = cap.read()
    labels = []
    # Mengkonversi frame yang diambil dari kamera menjadi grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Mendeteksi wajah dari frame
    faces = faceCascade.detectMultiScale(frame_gray, 1.3, 5)

    # Jika wajah ditemukan dalam frame, maka :
    for (x, y, w, h) in faces:
        # Membuat kotak di sekitar wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Mengambil citra wajah dari frame yang didapat
        face_detect = frame[int(y):int(y + h), int(x):int(x + w)]
        # Melakukan resize untuk menyesuaikan dengan ketentuan dari input dalam model
        face_detect = cv2.resize(face_detect, (224,224))

        # Jika face_detect memiliki data, maka :
        if np.sum([face_detect])!=0:
            # Mengubah gambar menjadi numpy array
            img_pixels = img_to_array(face_detect)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            # Melakukan prediksi dan menentukan label ke setiap gambar wajah
            predictions = model.predict(img_pixels)
            # Menampilkan hasil data prediksi ke terminal
            print("\nprediction = ", predictions)
            label = class_labels[predictions.argmax()]
            # Menampilkan label yang didapat dari prediksi
            print("\nprediction max = ", predictions.argmax())
            print("\nlabel = ", label)

            # Memberikan hasil deteksi di setiap kotak wajah
            cv2.putText(frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # Jika face_detect tidak memiliki data, maka :
        else:
            # Memberikan pemberitahuan bahwa, tidak ditemukannya wajah, 
            # karena var face_detect tidak memiliki nilai
            cv2.putText(frame, "No Face Found", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        print("\n\n")

    # Menampilkan video kamera
    cv2.imshow('Emotion Recognition', frame)
    # Jika mengklik 'q' maka akan mengeluarkan pengguna dari windo video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan layar
cap.release()
cv2.destroyAllWindoes()