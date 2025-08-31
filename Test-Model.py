import cv2
import mediapipe as mp
import pickle
import numpy as np

model_d = pickle.load(open('./model.p', 'rb'))
model = model_d['model']


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
mpDrawStyles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

labels = {0: 'C', 1: 'O'}
while True:
    ret, image = cap.read()
    data_setup = []

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                image,
                handLms,
                mp_hands.HAND_CONNECTIONS,
                mpDrawStyles.get_default_hand_landmarks_style(),
                mpDrawStyles.get_default_hand_connections_style()
            )
        
        for handLms in results.multi_hand_landmarks:
                for i in range(len(handLms.landmark)):
                    x = handLms.landmark[i].x
                    y = handLms.landmark[i].y
                    data_setup.append(x)
                    data_setup.append(y)

        prediction = model.predict([np.asarray(data_setup)])

        prediction_character = labels[int(prediction[0])]

        print(prediction_character)

    cv2.imshow("Image", image)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()