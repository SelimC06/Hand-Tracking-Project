import mediapipe as mp
import pickle
import cv2
import os
import matplotlib.pyplot as plt

DATA = "./data"

mp_hands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
mpDrawStyles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

data = []
label_data = []
for dir in os.listdir(DATA):
    for img_path in os.listdir(os.path.join(DATA, dir)):
        data_setup = []
        img = cv2.imread(os.path.join(DATA, dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for i in range(len(handLms.landmark)):
                    x = handLms.landmark[i].x
                    y = handLms.landmark[i].y
                    data_setup.append(x)
                    data_setup.append(y)
            data.append(data_setup)
            label_data.append(dir)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': label_data})
f.close()