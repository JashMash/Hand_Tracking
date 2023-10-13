import cv2 #For computer vision
import mediapipe as mp #For hand detection
import time

cap = cv2.VideoCapture(0)

# formality
mp_hands = mp.solutions.hands

hands= mp_hands.Hands()

# The draw function to see hand movements
mp_draw = mp.solutions.drawing_utils

while True:
    suc, img = cap.read()

    # Converting img
    img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            
            for id, lm in enumerate (hand_lms.landmark): # To get all hand landmark points
                h, w ,c = img.shape # height, width, channel
                cx, cy = int(lm.x * w), int()

                if id ==20:
                    cv2.circle(img, (cx,cy), 25, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)