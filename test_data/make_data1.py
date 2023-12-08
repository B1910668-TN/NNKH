import os
import cv2
import mediapipe as mp
import pandas as pd

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(1)

# Khởi tạo thư viện mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "oo"
no_of_frames = 600

# Tạo thư mục nếu chưa tồn tại
data_directory = "DATASET"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

def make_landmark_timestep(results):
    c_lm = []
    for hand_landmarks in results.multi_hand_landmarks:
        for id, lm in enumerate(hand_landmarks.landmark):
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    for hand_landmarks in results.multi_hand_landmarks:
        mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    return img

while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    if ret:
        # Nhận diện tay
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)

        if results.multi_hand_landmarks:
            # Ghi nhận thông số landmark của tay
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            # Vẽ landmark lên ảnh
            frame = draw_landmark_on_image(mpDraw, results, frame)

        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

# Ghi vào file
df  = pd.DataFrame(lm_list)
file_path = os.path.join(data_directory, f"{label}.txt")
df.to_csv(file_path)
cap.release()
cv2.destroyAllWindows()
