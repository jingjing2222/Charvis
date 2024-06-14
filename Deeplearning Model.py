import cv2
import numpy as np
import torch
from torchvision import transforms
from model import SimpleTSN  
import time
from picamera2 import Picamera2
import serial

# 시리얼 통신 포트와 속도 설정
ser = serial.Serial('/dev/ttyAMA4', baudrate=9600,timeout=1)

model = SimpleTSN(num_segments=5, num_classes=7)
model.load_state_dict(torch.load('tsn_best_model.pth', map_location=torch.device('cpu')))
model.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class_names = [1,2,3,4,5,6,7]


frame_queue = []
sequence_length = 5  


picam2 = Picamera2()
picam2.start()
time.sleep(2.0)

#for TSN
frame_queue = []


while True:
    #success, image = cap.read()
    image = picam2.capture_array()
    if image is None:
        break

    data_to_send = '0'
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_queue.append(frame_rgb)

    if len(frame_queue) > sequence_length:
        frame_queue.pop(0)

    if len(frame_queue) == sequence_length:
        batch = np.stack(frame_queue, axis=0)
        batch_tensor = torch.stack([transform(frame) for frame in batch])
        batch_tensor = batch_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            predictions = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            max_val, pred_class = torch.max(probabilities, 1)
            pred_class_name = class_names[pred_class.item()]
            confidence = max_val.item()
            data_to_send = pred_class_name
        # Reset frame_queue after processing
        frame_queue = []

        
        #flipped_image = cv2.flip(image, 1)


        
        #text = f'{pred_class_name}: {confidence:.2f}'
        #text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        #text_x = flipped_image.shape[1] - text_size[0] - 10  
        #text_y = 50  
        #cv2.putText(flipped_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        
        
        #cv2.imshow('MediaPipe Face Mesh', flipped_image)
        if data_to_send != 0:
            ser.write(data_to_send.to_bytes(2, byteorder='big'))  # 예를 들어 2바이트로 변환
            print("Sent:", data_to_send)
            time.sleep(1)  # 1초마다 데이터를 보내도록 설정
            
    if cv2.waitKey(5) & 0xFF == 27:
        
        break

cap.release()
#cv2.destoryAllWindows()
ser.close()
