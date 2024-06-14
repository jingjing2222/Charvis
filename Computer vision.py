import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms
from model import SimpleTSN  
import time
from picamera2 import Picamera2


YAWN_THRESHOLD = 20
EAR_THRESHOLD  = 0.21
base_face_size = 150

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14]

FOREHEAD_CENTER_INDEX = 10
CHIN_BOTTOM_INDEX = 152

def get_landmark_point(face_landmarks, landmark_index, image_shape):
    landmark_point = np.array([face_landmarks.landmark[landmark_index].x, face_landmarks.landmark[landmark_index].y]) * [image_shape[1], image_shape[0]]
    return landmark_point

def eye_aspect_ratio(eye_points):
    
    V1 = np.linalg.norm(eye_points[1] - eye_points[5])
    V2 = np.linalg.norm(eye_points[2] - eye_points[4])
    
    H = np.linalg.norm(eye_points[0] - eye_points[3])
    
    ear = (V1 + V2) / (2.0 * H)
    return ear


def calculate_lip_distance(face_landmarks, image_shape):
    upper_lip_point = np.array([face_landmarks.landmark[MOUTH_INDICES[0]].x, face_landmarks.landmark[MOUTH_INDICES[0]].y]) * [image_shape[1], image_shape[0]]
    lower_lip_point = np.array([face_landmarks.landmark[MOUTH_INDICES[1]].x, face_landmarks.landmark[MOUTH_INDICES[1]].y]) * [image_shape[1], image_shape[0]]
    distance = np.linalg.norm(upper_lip_point - lower_lip_point)
    return distance


def estimate_face_height(face_landmarks, image_shape):
    
    forehead_center_index = 10  
    
    chin_bottom_index = 152  

    
    forehead_center = get_landmark_point(face_landmarks, forehead_center_index, image_shape)
    chin_bottom = get_landmark_point(face_landmarks, chin_bottom_index, image_shape)

    
    face_height = np.linalg.norm(forehead_center - chin_bottom)
    
    return face_height


def adjust_thresholds(face_size):
    
    size_ratio = face_size / base_face_size

    adjusted_yawn_threshold = YAWN_THRESHOLD * size_ratio
    adjusted_ear_threshold = EAR_THRESHOLD * size_ratio

    return adjusted_yawn_threshold, adjusted_ear_threshold


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


picam2 = Picamera2()
picam2.start()
time.sleep(2.0)

#for TSN
frame_queue = []


with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    
    while True:
        #success, image = cap.read()
        image = picam2.capture_array()
        if image is None:
            break

        
        image.flags.writeable = False
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(image)

        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        yawn_status = "No Yawn Detected"  
        sleep_status = "Awake"  
        ear_text = "EAR: -"
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lip_distance = calculate_lip_distance(face_landmarks, image.shape)
                
                left_eye_points = np.array(
                    [get_landmark_point(face_landmarks, index, image.shape) for index in LEFT_EYE_INDICES])
                right_eye_points = np.array(
                    [get_landmark_point(face_landmarks, index, image.shape) for index in RIGHT_EYE_INDICES])
                left_EAR = eye_aspect_ratio(left_eye_points)
                right_EAR = eye_aspect_ratio(right_eye_points)
                ear = (left_EAR + right_EAR) / 2.0
                ear_text = f"EAR: {ear:.2f}"  
                
                if lip_distance > YAWN_THRESHOLD:
                    yawn_status = "Yawn Detected!"
                
                if ear < EAR_THRESHOLD:
                    sleep_status = "Sleepy"
                else:
                    sleep_status = "Awake"

                
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing
                    .DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                )
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        
                        face_height = estimate_face_height(face_landmarks, image.shape)
                        adjusted_yawn_threshold, adjusted_ear_threshold = adjust_thresholds(face_height)

                        
                        lip_distance = calculate_lip_distance(face_landmarks, image.shape)
                        left_eye_points = np.array(
                            [get_landmark_point(face_landmarks, index, image.shape) for index in LEFT_EYE_INDICES])
                        right_eye_points = np.array(
                            [get_landmark_point(face_landmarks, index, image.shape) for index in RIGHT_EYE_INDICES])
                        ear = (eye_aspect_ratio(left_eye_points) + eye_aspect_ratio(right_eye_points)) / 2

                        
                        yawn_status = "Yawn Detected!" if lip_distance > adjusted_yawn_threshold else "No Yawn Detected"
                        sleep_status = "Sleepy" if ear < adjusted_ear_threshold else "Awake"

        
        flipped_image = cv2.flip(image, 1)

        
        cv2.putText(flipped_image, yawn_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
            cv2.LINE_AA) 
        
        cv2.putText(flipped_image, sleep_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        
        cv2.putText(flipped_image, ear_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Face Mesh', flipped_image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destoryAllWindows()
