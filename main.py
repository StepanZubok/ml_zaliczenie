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
    img = cv2.resize(frame, (W, H))
    return np.expand_dims(img.astype("float32"), axis=0)


cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    # ret (return value):
    #   - True if frame captured successfully
    #   - False if error (camera disconnected, etc.)
    
    # frame:
    #   - Numpy array containing the image
    #   - Shape: (height, width, 3)
    #   - Example: (1080, 1920, 3) for Full HD

    ret, frame = cap.read()

    # Example successful read:
    # ret = True
    # frame = numpy.array([
    #     [[123, 89, 200], [124, 90, 201], ...],  # Row 1
    #     [[125, 88, 199], [126, 91, 202], ...],  # Row 2
    #     ...
    # ])  # Shape: (1080, 1920, 3)

    # # Example failed read (camera unplugged):
    # ret = False
    # frame = None  # or empty array
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