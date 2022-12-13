import cv2
import sys
import numpy as np
from time import sleep
from keras.models import load_model
from keras.utils import img_to_array
from keras.preprocessing import image

# Inisialisasi lokasi dari citra, cascade file, dan model file
imagePath = "./images/test3.jpg"
cascPath = "haarcascade_frontalface_default.xml"
modelPath = "./model/emotion_face_mobilNet.h5"
# imagePath = sys.argv[1]
# cascPath = sys.argv[2]

# Inisialisasi face cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Inisialisasi Model
model = load_model(modelPath)

# Inisialisasi Label
class_labels = ('Angry','Disgust','Fear', 'Happy','Neutral', 'Sad','Surprise')

# Membaca citra dan mengkonversinya menjadi greyscale
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Mendeteksi wajah di citra
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,                # scaleFactor, minNeighbor, dan minSize
    minNeighbors=5,                 # dapat dipermainkan sehingga mendapatkan faktor yang sesuai
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Membuat kotak di sekitar wajah
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Melakukan resize gambar untuk menyesuaikan dengan input model
# MobilNetv2: size = (48,48)
# Inception
image_predict = cv2.resize(image, (224,224)) 

# Mengubah gambar menjadi numpy array
img_pixels = img_to_array(image_predict)
img_pixels = np.expand_dims(img_pixels, axis=0)
img_pixels /= 255

# Melakukan prediksi dan menentukan label ke setiap gambar wajah
predictions = model.predict(img_pixels)
max_index = np.argmax(predictions[0])
class_labels = class_labels[max_index]

# Memberikan hasil deteksi di setiap kotak wajah
for (x, y, w, h) in faces:
    cv2.putText(image, class_labels, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# Menampilkan hasil deteksi
cv2.imshow("Faces found", image)
cv2.waitKey(0)
