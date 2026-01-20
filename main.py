import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

MODEL_PATH = "face_detector_02.keras"
THRESHOLD = 0.2
SAVE_DIR = "detections"

os.makedirs(SAVE_DIR, exist_ok=True)

model = load_model(MODEL_PATH)
H, W = model.input_shape[1], model.input_shape[2]   #mdel.input shape = (None, 256, 256, 3)

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H))
    return np.expand_dims(img.astype("float32"), axis=0)


cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    prediction = model.predict(preprocess(frame), verbose=0)[0][0]
    
    if prediction <= THRESHOLD:
        frame_count += 1
        filename = os.path.join(SAVE_DIR, f"face_{frame_count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"FACE SAVED: {filename} (conf: {prediction:.2%})")
    
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()