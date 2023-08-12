import cv2
import mediapipe as mp
import time
import math
import pyautogui

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

pTime = 0
cTime = 0

VOLUME_THRESHOLD = 65
SPEED_THRESHOLD = 70

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #thresh = (handLms.landmark[0].y*100 - handLms.landmark[9].y*100)/2
            thumb_tip_x, thumb_tip_y = int(handLms.landmark[4].x * img.shape[1]), int(handLms.landmark[4].y * img.shape[0])
            index_tip_x, index_tip_y = int(handLms.landmark[8].x * img.shape[1]), int(handLms.landmark[8].y * img.shape[0])
            distance = calculate_distance(thumb_tip_x, thumb_tip_y, index_tip_x, index_tip_y)
            cv2.line(img, (thumb_tip_x, thumb_tip_y), (index_tip_x, index_tip_y), (0, 255, 0), 5)
            
            if distance< VOLUME_THRESHOLD:
                hand_gesture = "vol_down"

            elif distance > VOLUME_THRESHOLD:
                hand_gesture = "vol_up"
            else:
                hand_gesture = "other"
        
            if hand_gesture =="vol_up":
                pyautogui.press('volumeup')

            elif hand_gesture =="vol_down":
                pyautogui.press("volumedown")
              
        
            if distance < VOLUME_THRESHOLD:
                cv2.putText(img, f"Distance:{round(distance,2)}<65, Volume Decreasing", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, f"Distance:{round(distance,2)}>65, Volume Increasing", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0, 255), 2)

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f" {str(int(fps))} FPS", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()