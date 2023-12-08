import cv2
from ultralytics import YOLO
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import time
import cv2
import mediapipe as mp
import tensorflow as tf
import threading

# YOLOv8 và LSTM models
model = YOLO("test_model\weights\\best.pt")
model_lstm = tf.keras.models.load_model("model.h5")

# Kết nối với webcam
cap = cv2.VideoCapture(1)  

# Font
font_path = 'font\mdright.ttf' 

pil_font = ImageFont.truetype(font_path, size=45) 
bounding_box_font = ImageFont.truetype(font_path, size=30)

# Đặt vị trí font YOLOv8
bottom_right_corner = (10, 400)  
label_count = 0  
all_labels = []  
start_time = time.time() 

# Đặt vị trí font LSTM
label_lstm = '....'
n_time_steps = 30
lm_list = []

# MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

def make_landmark_timestep(results_lstm):
    c_lm = []
    if results_lstm.multi_hand_landmarks:
        for hand_landmarks in results_lstm.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                c_lm.append(lm.x)
                c_lm.append(lm.y)
                c_lm.append(lm.z)
                c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results_lstm, frame):
    if results_lstm.multi_hand_landmarks:
        for hand_landmarks in results_lstm.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return frame

def draw_class_on_image(label_lstm, frame):
    bottomLeftCornerOfText = (10, 25)
    fontColor = ("blue")
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame)
    draw.text(bottomLeftCornerOfText, label_lstm, font=pil_font, fill=fontColor)
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    return frame


def detect(model_lstm, lm_list):
    global label_lstm
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model_lstm.predict(lm_list)
    print(results)
    
    # Lấy chỉ số của lớp có giá trị lớn nhất
    predicted_class = np.argmax(results, axis=1)
    
    label_names = ["Dấu sắc", "Dấu huyền", "Chanh muối", "Chanh đá", "Xin chào", "Tôi yêu bạn", "Ă", "Â", "Ê", "Ô", "Ư"]
    label_lstm = label_names[predicted_class[0]]
    
    return label_lstm
        
while True:
    ret, frame = cap.read()

    if ret:
        # Dự đoán với model YOLOv8
        results = model.predict(frame, show=False)

        result = results[0]
        
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_frame)
        
        
        for box in result.boxes:
            label = result.names[box.cls[0].item()]
            conf = round(box.conf[0].item(), 2)
            
            # Kiểm tra xem label đã xuất hiện trước đó hay chưa
            if label not in all_labels:
                all_labels.append(label)
                # Cập nhật số lượng label đã vẽ
                label_count += 1
            
            # Lấy tọa độ của hộp giới hạn
            x1, y1, x2, y2 = box.xyxy[0]
            
            # Vẽ bounding box bằng hàm rectangle của ImageDraw
            draw.rectangle([(x1, y1), (x2, y2)], outline=("red"), width=2)
            
            # Vẽ background cho nhãn
            text_bbox = draw.textbbox((x1, y1 - 43), f"{label} {conf}", font=bounding_box_font)
            
            expanded_bbox = (
                text_bbox[0],
                text_bbox[1],
                text_bbox[2] + 10,
                text_bbox[3] + 10
            )
            
            draw.rectangle(expanded_bbox, fill="red")
            
            # Vẽ label lên bounding box
            draw.text((x1 + 3, y1 - 38), f"{label} {conf}", font=bounding_box_font, fill="white")  

        # Hiển thị label ở góc dưới trái màn hình
        label_text = ' - '.join(all_labels)
        draw.text(bottom_right_corner, label_text, font=pil_font, fill="white")

        # Nếu đã hiển thị trong 5 giây, reset lại
        if time.time() - start_time >= 5:
            all_labels = []
            label_count = 0  # Reset số lượng label đã vẽ
            start_time = time.time()  # Reset thời gian bắt đầu hiển thị
            
        # Chuyển đổi lại sang định dạng mảng NumPy để hiển thị
        frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

        # Xử lý cho model LSTM
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        results_lstm = hands.process(imgRGB)
        if results_lstm.multi_hand_landmarks:
            c_lm = make_landmark_timestep(results_lstm)
            lm_list.append(c_lm)
            
            if len(lm_list) == n_time_steps:
                t1 = threading.Thread(target=detect, args=(model_lstm, lm_list,)) #Tạo một xử lý đa luồng
                t1.start()
                lm_list = []
                
            frame = draw_landmark_on_image(mpDraw, results_lstm, frame)
               
        frame = draw_class_on_image(label_lstm, frame)             
        
        cv2.imshow('Demo', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
