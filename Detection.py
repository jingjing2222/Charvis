# CO2, model, mp 연결 완료
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from picamera2 import Picamera2
import serial
from dotenv import load_dotenv
import os
import pyaudio
import speech_recognition as sr
import wave
from openai import OpenAI
import bluetooth
# 졸음 상태 저장용 배열 (하품, 눈 깜빡임 지속 시간, 분당 눈 깜빡임 횟수, 분당 눈 감은 시간)
drowsy_weight = [0, 0, 0, 0, 0]  # CO2(가중치) 하품(가중치 : 1 or 0), 눈 깜빡임 지속시간(가중치), 분당 눈 깜빡임 횟수, 분당 눈 감은 시간
co2_level = 0
drowy_level = 1

load_dotenv()  # .env 파일에서 환경 변수 불러오기

def setup_bluetooth_connection():
    # Bluetooth 서버 설정
    server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_socket.bind(("", bluetooth.PORT_ANY))
    server_socket.listen(1)

    print("Bluetooth 연결을 기다리는 중...")
    client_socket, address = server_socket.accept()
    print(f"{address} 에 연결되었습니다.")

    # Bluetooth 연결이 완료되면 사용자 정보 수신
    user_interest = []
    while True:
        data = client_socket.recv(1024).decode('utf-8')
        if data:
            read_message = data.split("|")
            if read_message[0] == "0":
                print(f"유저 정보: {read_message[1]}, {read_message[2]}")
                user_interest.append(read_message[1])
                user_interest.append(read_message[2])
                break

    return client_socket, user_interest


class Charvis:
    def __init__(self):
        self.system_messages = {
            "start": """ 
            Charvis는 운전자가 깨어 있도록 설계된 대화형 서비스입니다. 간단한 자기 소개로 대화를 시작합니다 아래의 국토교통부 선정 졸음운전 예방 8가지 수칙 중 한 가지를 언급합니다.
            1. 하루 7~8시간 충분한 수면 취하기
            2. 충분한 수면 휴식 취했는데도 졸음이 오면 반드시 다른 요인이 있는지 살펴야 한다.
            3. 차내 이산화탄소 농도가 높아지면 산소부족으로 졸음이 발생하므로 주기적으로 창문을 열어 내부 환기 시키기
            4. 졸음을 유발하는 성분이 든 약물은 운전할때 가급적 삼가기
            5. 휴게소나 졸음 쉼터에서 쉬어가기
            6. 운전하는 날은 졸음을 유발하는 음식물 피하기
            7. 졸음방지 용품 적극적으로 활용해보기
            8. 간단한 스트레칭이나 지압해주기
            졸음 레벨, CO2 레벨은 Charvis가 측정한 결과입니다. 이모티콘을 사용하지 않습니다.
            졸음 레벨을 직접적으로 언급하지 않습니다.
            """,
             "level_1": """
            사용자가 졸음이 얼추 가셨습니다., 흥미에 대한 질문을 생성합니다, 한국어로만 대화를 생성합니다.
            """,
            "level_2": """
            사용자가 졸음이 오고 있는 단계입니다, 흥미에 대한 질문을 생성합니다, 한국어로만 대화를 생성합니다.

            """,
            "level_3": """
            사용자가 매우 심한 졸음을 느끼고 있는 단계입니다, 사용자에게 졸음쉼터로 안내해도 되는지 질문을 생성합니다. 사용자에게 안내를 원하신다면 안내해줘라고 말해야 한다고 언급합니다. 해당 질문을 생성한 후에도 졸음쉼터로 안내한 후에도 대화는 계속됩니다.
            """,
            "level_3_1": """
            사용자가 매우 심한 졸음을 느끼고 있는 단계입니다, 사용자와의 대화를 지속합니다. 흥미에 대한 질문을 생성합니다.
            """,
            "level_4": """
            사용자가 거의 자고 있다고 봐도 무방한 상태입니다. 사용자가 위험한 상태이니 사랑하는 가족들을 위해 당장 졸음을 깨달라는 말을 해주세요.
            """,
            "center_guide": """
            사용자가 졸음 쉼터로 네비게이션을 조정하는 것을 동의하였거나 인지했다고 판단됨 졸음 쉼터로 네비게이션을 조정한다는 말 생성함. 생성 후에도 흥미에 관한 대화는 지속됨.
            """,
            "end": """
            사용자에게 안전운전 하라는 말과 작별인사를 하고 대화를 종료합니다.
            """
        }
        self.prev_question = None
        self.conversation_count = 0
        self.interest = None
        self.iscenterquestion = 0
        self.iscentergoing=0

    def is_speech(self, data, threshold):
        return np.max(np.frombuffer(data, dtype=np.int16)) > threshold

    def capture_voice(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 4096
        SILENCE_THRESHOLD = 3
        SPEECH_THRESHOLD = 2000

        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("음성 인식을 시작합니다. 말씀해주세요.")
        frames = []
        silence_counter = 0

        while True:
            data = stream.read(CHUNK)
            frames.append(data)

            if self.is_speech(data, SPEECH_THRESHOLD):
                silence_counter = 0
            else:
                silence_counter += 1

            if silence_counter > SILENCE_THRESHOLD * RATE / CHUNK:
                break

        print("음성 인식이 완료되었습니다.")
        wf = wave.open("temp.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        r = sr.Recognizer()
        with sr.AudioFile("temp.wav") as source:
            audio_data = r.record(source)
            try:
                text = r.recognize_google(audio_data, language='ko-KR')
                print(f"인식된 텍스트: {text}")
                return text
            except sr.UnknownValueError:
                print("음성을 인식할 수 없습니다.")
                return None
            except sr.RequestError as e:
                print(f"STT 서비스에 접근할 수 없습니다. 오류: {e}")
                return None
            finally:
                stream.stop_stream()
                stream.close()
                audio.terminate()

    def create_prompt(self, sleep_level, co2_level, user_interest, user_voice=None):
        self.interest = user_interest
        prompt = f"졸음 레벨: {sleep_level}, CO2 레벨: {co2_level}, 흥미: {self.interest}, 대화 횟수: {self.conversation_count}"
        if user_voice:
            prompt += f", 사용자 음성 입력: '{user_voice}'"
        else:
            prompt += ", 사용자 음성 입력: None"

        if self.prev_question:
            prompt += f"\n이전 질문: {self.prev_question}\n이전 답변: {user_voice}"

        return prompt

    def create_role(self, sleep_level, user_voice=None):
        if int(sleep_level) >= 2:
            if user_voice:
                role = "사용자가 응답했습니다."
            else:
                role = "사용자의 졸음 레벨이 높게 감지되었습니다."
        else:
            role = "사용자가 응답했습니다."
        return role

    def call_gpt_api(self, prompt, system_message):
        client = OpenAI()
        OpenAI.api_key = os.getenv("OPENAI_API_KEY")

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.05,
        )

        generated_text = response.choices[0].message.content
        return generated_text

    def extract_flags(self, user_voice):
        center_flag = 0
        if self.iscenterquestion == 1:
            positive_answers = ["안내해줘","안 내 해 줘","안 내해줘","안내해 줘","안 내 해줘","안내 해줘","안내"]
            if user_voice is not None:
                for answer in positive_answers:
                    if answer in user_voice:
                        center_flag = 1
                        break
        return center_flag

    def generate_prompt(self, sleep_level, co2_level, user_interest, user_voice):
        center_flag = self.determine_center_flag(sleep_level, user_voice)
        if self.conversation_count == 0:
            system_message = self.system_messages["start"]
            prompt = self.create_prompt(sleep_level, co2_level, user_interest, user_voice)
        elif int(sleep_level) == 1:
            system_message = self.system_messages["level_1"]
            prompt = self.create_prompt(sleep_level, co2_level, user_interest, user_voice)
        elif int(sleep_level) == 2:
            system_message = self.system_messages["level_2"]
            prompt = self.create_prompt(sleep_level, co2_level, user_interest, user_voice)
        elif int(sleep_level) == 3 and self.iscenterquestion == 0:
            system_message = self.system_messages["level_3"]
            prompt = self.create_prompt(sleep_level, co2_level, user_interest, user_voice)
            self.iscenterquestion = 1

        elif int(sleep_level) == 3 and self.iscenterquestion == 1 and self.iscentergoing == 0 and center_flag == 0:
            system_message = self.system_messages["level_3_1"]
            prompt = self.create_prompt(sleep_level, co2_level, user_interest, user_voice)
            self.iscenterquestion = 1

        elif int(sleep_level) == 3 and self.iscenterquestion == 1 and center_flag == 1:
            system_message = self.system_messages["center_guide"]
            prompt = self.create_prompt(sleep_level, co2_level, user_interest, user_voice)
            self.iscenterquestion = 1
            self.iscentergoing=0

        elif int(sleep_level) == 4:
            print("sleep_level:",sleep_level)
            system_message = self.system_messages["level_4"]
            prompt = self.create_prompt(sleep_level, co2_level, user_interest, user_voice)

        if int(co2_level) >= 2:
            if int(co2_level) == 2:
                system_message += "\n현재 CO2 농도가 조금 높습니다. 환기를 제안합니다."
            elif int(co2_level) == 3:
                system_message += "\n현재 CO2 농도가 높습니다. 환기를 제안합니다."
            elif int(co2_level) == 4:
                system_message += "\n현재 CO2 농도가 매우 높습니다. 즉시 환기를 강요합니다."
        return system_message, prompt

    def determine_center_flag(self, sleep_level, user_voice):
        if int(sleep_level) == 3:
            center_flag = self.extract_flags(user_voice)
        else:
            center_flag = 0
        return center_flag

    def send_response(self, prompt, sleep_level, center_flag, generated_response, mThreadConnectedBluetooth):
        print(f"\n전송된 프롬프트: {prompt}\n")
        generated_question = generated_response.split('{')[0].strip()
        generated_question = f"{sleep_level}|{center_flag}|{generated_question}"
        print(f"GPT API가 생성한 응답: {generated_question}\n")
        mThreadConnectedBluetooth.send(generated_question.encode('utf-8'))

    def get_levels(self):
        global co2_level, drowy_level
        sleep_level = str(drowy_level)
        co2_level_ = str(co2_level)
        return sleep_level, co2_level_

    def wait_for_user_input(self, mThreadConnectedBluetooth):
        print("안드로이드에서 시간 값을 기다리는 중...")
        while True:
            try:
                t = mThreadConnectedBluetooth.recv(1024).decode('utf-8')
                break
            except:
                pass
        print(f"받아온 대기 시간: {t}초")
        time.sleep(int(t))
        user_voice = self.capture_voice()
        return user_voice

    def run(self, client_socket, user_interest):
        test_bool = True
        count = 0
        
        while True:
            #sleep_level, co2_level = self.get_levels()
            sleep_level=input("sleep 레벨을 입력하세요: ")
            co2_level=input("co2 레벨을 입력하세요: ")
            if int(sleep_level) <= 1:
                print("졸음 레벨이 1 이하입니다. 10초 후에 다시 확인합니다.")
                time.sleep(10)
                continue

            while int(sleep_level) >= 2:
                #sleep_level, co2_level = self.get_levels()
                if self.conversation_count == 0:
                    user_voice = None
                else:
                    user_voice = self.wait_for_user_input(client_socket)

                system_message, prompt = self.generate_prompt(sleep_level, co2_level, user_interest[1], user_voice)
                center_flag = self.determine_center_flag(sleep_level, user_voice)
                generated_response = self.call_gpt_api(prompt, system_message)
                self.send_response(prompt, sleep_level, center_flag, generated_response, client_socket)

                self.prev_question = generated_response.split('{')[0].strip()
                self.conversation_count += 1

                #sleep_level, co2_level = self.get_levels()
                sleep_level=input("sleep 레벨을 입력하세요: ")
                co2_level=input("co2 레벨을 입력하세요: ")

                # sleep_level이 2 이상이다가, 1로 떨어질 경우, 안전운전하라는 말과 함께 대화를 종료함
                if int(sleep_level) == 1:
                    test_bool = False
                    count += 1
                    if count == 3:
                        print("안드로이드에서 시간 값을 기다리는 중...")
                        while True:
                            try:
                                t = client_socket.recv(1024).decode('utf-8')
                                break
                            except:
                                pass
                        print(f"받아온 대기 시간: {t}초")
                        time.sleep(int(t))
                        system_message = self.system_messages["end"]
                        prompt = self.create_prompt(sleep_level, co2_level, user_interest[1], None)
                        center_flag = self.determine_center_flag(sleep_level, user_voice)
                        generated_response = self.call_gpt_api(prompt, system_message)
                        self.send_response(prompt, sleep_level, center_flag, generated_response, client_socket)
                        print("졸음 레벨이 1 이하로 떨어졌습니다. 10초 후에 다시 확인합니다.")
                        time.sleep(10)
                        break



# 0 : 좋음, 1 : 평범, 2 : 주의, 3 : 피로유발, 4 : 위험 - 태경
class CO2Sensor:
    def __init__(self, port='/dev/ttyAMA0', baudrate=9600):
        self.ser = serial.Serial(port, baudrate)
        self.receive_buff = [0] * 8
        self.level = 0  # 레벨 유지
        self.pre_level = 0
        self.weight = 0

    # 초기화 작업
    def setup(self):
        self.ser.flushInput()
        self.ser.flushOutput()
        time.sleep(2)

    # CO2 센서로 데이터 요청 명령 전송
    def send_cmd(self):
        send_data = [0x11, 0x01, 0x01, 0xED]
        for data in send_data:
            self.ser.write(bytes([data]))
            time.sleep(0.001)

    # 데이터의 체크섬 계산
    def checksum_cal(self):
        SUM = sum(self.receive_buff[:7])
        checksum = (256 - SUM % 256) % 256
        return checksum

    # ppm 값에 따라 CO2 상태 분류
    def co2_level(self, ppm):
        if ppm <= 1000:
            # 0단계
            return 0  # 좋음
        elif ppm <= 2000:
            # 1단계
            return 1  # 평범
        elif 1000 < ppm <= 2500:
            # 2단계
            return 2  # 주의 - 환기 or 외기순환 제안
        elif 2500 < ppm <= 3000:
            # 3단계
            return 3  # 피로 유발 - 환기 or 외기순환 제안
        else:
            # 4단계
            return 4  # 위험 - 환기 or 외기순환 제안

    def co2_weight(self):
        if self.level == 0:
            # 0단계
            self.level = 0
            return 0
        elif self.level == 1:
            # 1단계
            return 0.3
        elif self.level == 2:
            # 2단계
            return 0.5
        elif self.level == 3:
            # 3단계
            return 0.8
        elif self.level <= 4:
            # 4단계
            return 1.0

    def co2_check(self):
        global drowsy_weight, co2_level
        self.setup()
        print("Sending")
        self.send_cmd()
        time.sleep(1)

        recv_cnt = 0
        time.sleep(1)
        # 센서로부터 8바이트의 데이터 수신
        while recv_cnt < 8:
            if self.ser.in_waiting > 0:
                self.receive_buff[recv_cnt] = ord(self.ser.read(1))
                recv_cnt += 1
        # 수신된 데이터의 체크섬 검사
        if self.checksum_cal() == self.receive_buff[7]:
            PPM_Value = (self.receive_buff[3] << 8) | self.receive_buff[4]
            # 레벨 반환
            self.level = self.co2_level(PPM_Value)  # 0,1,2,3,4
            # 이전 레벨 < 현재 ppm -> 이전레벨 + 1
            if self.pre_level < self.level:
                self.level = self.pre_level + 1
            # 가중치 반환
            self.weight = self.co2_weight()
            # 이전 레벨 저장
            self.pre_level = self.level
            print(self.level)
            co2_level, drowsy_weight[0] = self.level, self.weight
        else:
            print("CHECKSUM Error 이전 레벨로 입력됩니다.")
            co2_level, drowsy_weight[0] = self.level, self.weight

    # CO2 상태를 지속적으로 확인하는 루프
    def run(self):
        self.co2_check()
        time.sleep(20)  # 15분 뒤부터 코드 동작
        while True:
            self.co2_check()
            time.sleep(20)  # 10분마다 코드 동작


# Serial port settings
ser = serial.Serial('/dev/ttyAMA4', baudrate=9600, timeout=1)
sensor = CO2Sensor()

# MediaPipe 설정
mp_drawing = mp.solutions.drawing_utils  # cv출력용
mp_drawing_styles = mp.solutions.drawing_styles  # cv출력용
mp_face_mesh = mp.solutions.face_mesh
# 얼굴의 관심 영역 인덱스 (눈, 입 등)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14]
# 눈 깜빡임, 하품 검출 기준값
BLINK_RESET_TIME = 60
EAR_THRESHOLD = 0.15  # 눈 깜빡임 기준값
YAWN_THRESHOLD = 20  # 하품 감지 기준값

# 시리얼 통신을 통한 데이터 수신
received_int = None


def serial_read():
    global received_int
    while True:
        # 2바이트 데이터 수신
        data = ser.read(2)
        if data:
            received_int = int.from_bytes(data, byteorder='big')
            #print("Received:", received_int)


class DrowsinessDetector:
    def __init__(self):
        # 졸음 감지에 필요한 초기값 설정
        self.blink_count = 0
        self.start_time = time.time()
        self.blink_timestamp = 0
        self.closing_durations = []
        self.eye_frame_count = 0
        self.mouth_frame_count = 0
        self.last_blink_timestamp = 0
        self.blinks_per_minute = []
        self.closed_eye_time = 0
        self.start_measurement_time = time.time()
        self.eye_closed_timestamp = 0
        # 하품
        self.yawn_start_time = 0  # 시작시간
        self.yawn_duration = 0  # 기간
        self.yawn_count = 0  # 횟수
        self.stage = 0  # 상태
        self.stage_start_time = time.time()  # 시간
        self.weight_yawming = 0
        # 지속시간
        self.weight_eye_close_time = 0
        # 횟수
        self.weight_blink_count = 0
        # 퍼클로스
        self.weight_perclos = 0

    ## 하품 - 태경
    def detect_yawning(self, face_landmarks, image_shape):
        # 입 크기 계산
        upper_lip_point = np.array(
            [face_landmarks.landmark[MOUTH_INDICES[0]].x, face_landmarks.landmark[MOUTH_INDICES[0]].y]) * [
                              image_shape[1], image_shape[0]]
        lower_lip_point = np.array(
            [face_landmarks.landmark[MOUTH_INDICES[1]].x, face_landmarks.landmark[MOUTH_INDICES[1]].y]) * [
                              image_shape[1], image_shape[0]]
        lip_distance = np.linalg.norm(upper_lip_point - lower_lip_point)
        # print(lip_distance)
        # 하품 감지 및 지속 시간 처리
        if lip_distance > YAWN_THRESHOLD:
            if self.yawn_start_time == 0:  # 하품이 시작된 시간이 기록되지 않았다면
                self.yawn_start_time = time.time()  # 현재 시간을 하품 시작 시간으로 기록
            else:
                self.yawn_duration = time.time() - self.yawn_start_time  # 하품 지속 시간 계산
                if self.yawn_duration >= 2.5:  # 하품이 3초 이상 지속되었다면
                   self.yawn_count += 1
                   self.weight_yawming = self.yawning_update_stage()  # 0, 1
                   self.yawn_start_time = 0
                   self.yawn_duration = 0
        else:
            # 하품이 감지되지 않으면 시작 시간과 지속 시간 리셋
            self.yawn_start_time = 0
            self.yawn_duration = 0
            self.weight_yawming = self.yawning_update_stage()
        return self.weight_yawming

    # 하품 가중치
    def yawning_update_stage(self):
        current_time = time.time()  # 실시간
        elapsed_time = current_time - self.stage_start_time  ## 단계 시작 시간
        old_stage = self.stage

        if self.stage == 0 and self.yawn_count >= 1:
            self.stage = 1
        elif self.stage == 1 and elapsed_time >= 300:  # 1단계 : 5분간 지속
            if self.yawn_count == 0:  # 5분간 0회 시 0단계 격하
                self.stage = 0
            elif self.yawn_count == 1:  # 5분간 1회 시 1단계 유지
                self.stage = 1
            elif self.yawn_count == 2:  # 5분간 2회 시 2단계 격상
                self.stage = 2
            elif self.yawn_count >= 3:  # 5분간 3회 시 3단계 격상
                self.stage = 3
        elif self.stage == 2 and elapsed_time >= 900:  # 2단계 : 15분간 지속
            if self.yawn_count <= 1:
                self.stage = 1
            elif self.yawn_count == 2:
                self.stage = 2
            elif self.yawn_count >= 3:
                self.stage = 3
        elif self.stage == 3 and elapsed_time >= 1800:  # 3단계 : 30분간 지속
            if self.yawn_count <= 1:
                self.stage = 1
            elif self.yawn_count == 2:
                self.stage = 2
            elif self.yawn_count >= 3:
                self.stage = 3

        if old_stage != self.stage:  # 단계가 변경된 경우
            self.stage_start_time = current_time
            self.yawn_count = 0  # 단계가 변경되면 하품 횟수 초기화
            if self.stage != 0:
                return 1  # 단계가 0이 아닐 때 가중치 1 반환
            else:
                return 0  # 단계가 0일 때 0 반환
        else:  # 변경이 되지 않았다면
            if self.stage != 0:
                return 1  # 단계가 0이 아닐 때 가중치 1 반환
            else:
                return 0  # 단계가 0일 때 0 반환

    # 눈의 EAR 계산 함수 - 태경
    def eye_aspect_ratio(self, eye_points):
        V1 = np.linalg.norm(eye_points[1] - eye_points[5])
        V2 = np.linalg.norm(eye_points[2] - eye_points[4])
        H = np.linalg.norm(eye_points[0] - eye_points[3])
        ear = (V1 + V2) / (2.0 * H)
        return ear

    # EAR 계산을 통한 졸음 감지 - 태경
    def EAR_calculation(self, face_landmarks, image_shape):
        left_eye_points = np.array([np.array([face_landmarks.landmark[index].x, face_landmarks.landmark[index].y]) * [
            image_shape[1], image_shape[0]] for index in LEFT_EYE_INDICES])
        right_eye_points = np.array([np.array([face_landmarks.landmark[index].x, face_landmarks.landmark[index].y]) * [
            image_shape[1], image_shape[0]] for index in RIGHT_EYE_INDICES])
        left_ear = self.eye_aspect_ratio(left_eye_points)
        right_ear = self.eye_aspect_ratio(right_eye_points)

        min_ear = min(left_ear, right_ear)

        return min_ear

    def calculate_weight(self):
        drowsy_count = len([d for d in self.closing_durations if d >= 0.5])
        base_weight = None
        # 횟수에 따른 기본 가중치 설정
        if drowsy_count == 0:
            base_weight = 0
        elif drowsy_count == 1:
            base_weight = 0.1
        elif drowsy_count == 2:
            base_weight = 0.3
        elif drowsy_count >= 3:
            base_weight = 0.6
        # 횟수에 따른 추가 가중치 설정
        if drowsy_count < 3:
            additional_weight = 0
        elif drowsy_count < 5:
            additional_weight = 0.2
        else:
            additional_weight = 0.3

        self.weight_eye_close_time = round(base_weight + additional_weight,1)
        #print(f"지속시간 가중치 : {self.weight_eye_close_time}, 횟수 :{drowsy_count}")
        return self.weight_eye_close_time

    # 눈 지속 시간 계산 - 태경
    def calculate_eye_closing_time(self, ear):
        current_time = time.time()
        # 눈의 개방 정도(ear)가 임계값(EAR_THRESHOLD)보다 작고, 눈을 감기 시작한 시간이 기록되지 않았다면
        if ear < EAR_THRESHOLD and self.blink_timestamp == 0:
            self.blink_timestamp = current_time  # 현재 시간을 눈 감기 시작 시간으로 설정
        # 눈의 개방 정도가 임계값 이상이고, 눈을 감기 시작한 시간이 기록되어 있다면 (즉, 눈이 다시 열렸다면)
        elif ear >= EAR_THRESHOLD and self.blink_timestamp != 0:
            duration = current_time - self.blink_timestamp  # 눈을 감고 있던 총 시간 계산
            if len(self.closing_durations) >= 15:
                self.closing_durations.pop(0)  # 리스트의 크기를 15으로 유지하기 위해 가장 오래된 기록 삭제
                #print("!")
            self.closing_durations.append(duration)  # 새로운 눈 감김 지속 시간을 리스트에 추가
            self.blink_timestamp = 0  # 눈 감기 시작 시간 초기화
            self.blink_count += 1  # 눈 깜박임 카운트 증가
            return self.calculate_weight()
        return self.weight_eye_close_time

    # 분당 눈 깜박임 횟수 계산 - 태경
    def calculate_blink_count_and_rate(self):
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time >= 60:
            # 1분간의 눈 깜박임 횟수가 5회 미만이거나 20회 초과하면 1 추가
            if self.blink_count < 5 or self.blink_count > 20:
                self.blinks_per_minute.append(1)
            else:
                self.blinks_per_minute.append(0)  # 최근 1분간 깜박임 수를 리스트에 추가

            self.blink_count = 0  # 카운터 리셋
            self.start_time = current_time  # 시간 리셋

            if len(self.blinks_per_minute) > 5:  # 최근 5분 기록만 유지
                self.blinks_per_minute.pop(0)

            # 최근 5분간 총 깜박임 횟수 계산
            total_blinks = sum(self.blinks_per_minute)

            # 가중치 계산
            if total_blinks == 0:
                self.weight_blink_count = 0
            elif total_blinks == 1:
                self.weight_blink_count = 0.5
            elif total_blinks == 2:
                self.weight_blink_count = 0.8
            elif total_blinks >= 3:
                self.weight_blink_count = 1.0

        return self.weight_blink_count

    # 눈 깜빡임 상태를 업데이트하는 함수 - 태경
    def update_eye_closure(self, ear):
        current_time = time.time()
        if ear < EAR_THRESHOLD:
            # 눈이 감기 시작했을 때 타임스탬프 기록
            if self.eye_closed_timestamp == 0:  # 눈이 감기 시작했을 때
                self.eye_closed_timestamp = current_time
        else:
            # 눈이 다시 열렸을 때 닫힌 눈의 시간을 계산하고 타임스탬프 초기화
            if self.eye_closed_timestamp != 0:  # 눈이 다시 열렸을 때
                self.closed_eye_time += current_time - self.eye_closed_timestamp
                self.eye_closed_timestamp = 0

    # PERCLOS (눈 감긴 상태 비율) 계산 함수 - 태경
    def calculate_perclos(self):
        current_time = time.time()
        # 60초마다 PERCLOS 계산
        if current_time - self.start_measurement_time >= 60:
            total_time = current_time - self.start_measurement_time
            perclos = (self.closed_eye_time / total_time) * 100
            # 측정 시간 및 닫힌 눈의 시간 리셋
            self.start_measurement_time = current_time
            self.closed_eye_time = 0
            # PERCLOS가 20% 미만인지 확인하여 상태 반환
            if perclos < 20:
                self.weight_perclos = 0
            elif perclos < 30:
                self.weight_perclos = 0.6
            elif perclos < 40:
                self.weight_perclos = 1.0
            elif perclos >= 40:
                self.weight_perclos = 1.5
        return self.weight_perclos  # 60초가 지나지 않았다면 이전 상태 반환


# 졸음 감지기 인스턴스 생성 - 기본
detector = DrowsinessDetector()
# 차비스 생성
charvis = Charvis()


# 얼굴 랜드마크의 좌표를 반환하는 함수 - 기본
def get_landmark_point(face_landmarks, landmark_index, image_shape):
    return np.array([face_landmarks.landmark[landmark_index].x, face_landmarks.landmark[landmark_index].y]) * [
        image_shape[1], image_shape[0]]


# 눈의 EAR를 계산하는 함수 - 태경
def eye_aspect_ratio(eye_points):
    V1 = np.linalg.norm(eye_points[1] - eye_points[5])
    V2 = np.linalg.norm(eye_points[2] - eye_points[4])
    H = np.linalg.norm(eye_points[0] - eye_points[3])
    return (V1 + V2) / (2.0 * H)

class TextStatus:
    def __init__(self):
        self.co2_status = ""
        self.drowsy0_status = ""
        self.drowsy1_status = ""
        self.drowsy2_status = ""
        self.drowsy3_status = ""
        self.drowsy_status = ""
        self.received_int=""
    def update_status(self, drowsy_weight, ):
        self.co2_status = f"co2: {drowsy_weight[0]:.2f}"
        self.drowsy0_status = f"Yawning: {drowsy_weight[1]:.2f}"
        self.drowsy1_status = f"Blink Time: {drowsy_weight[2]:.2f}"
        self.drowsy2_status = f"Blink Rate: {drowsy_weight[3]:.2f}"
        self.drowsy3_status = f"PERCLOS: {drowsy_weight[4]:.2f}"
        self.drowsy4_status = f"sleep level: {drowy_level}"
        self.received_int = f"model classification: {received_int}"

def add_text_with_background(image, text, position, font, font_scale, text_color, thickness, background_color):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    baseline += thickness
    top_left = (position[0], position[1] - text_height - baseline)
    bottom_right = (position[0] + text_width, position[1] + baseline)
    cv2.rectangle(image, top_left, bottom_right, background_color, -1)
    cv2.putText(image, text, position, font, font_scale, text_color, thickness, cv2.LINE_AA)

# 메인 함수: 카메라 영상 및 얼굴 메쉬 감지 루프
def drowy_run():
    global drowy_level, received_int
    text_status = TextStatus()
    picam2 = Picamera2()
    picam2.start()
    # MediaPipe 얼굴 메쉬 감지 설정
    last_print_time = time.time()  # 마지막으로 drowy_level을 출력한 시간을 저장

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while True:
            image = picam2.capture_array()
            if image is None:
                break
            # 이미지를 RGB로 변환 후, 얼굴 메쉬 감지
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # ----------------------------------------------------------------------------------------
                    # 하품
                    # print(drowsy_weight[0])
                    # 입 크기 계산 - 실시간 업데이트
                    drowsy_weight[1] = detector.detect_yawning(face_landmarks, image.shape)
                    # print(drowsy_weight[1])
                    # ----------------------------------------------------------------------------------------
                    # 눈 깜빡임 지속시간 - 실시간 업데이트
                    # ear 계산
                    ear = detector.EAR_calculation(face_landmarks, image.shape)
                    # 눈 깜빡임 시 눈 감김 지속 시간이 500ms을 넘으면 상태 True로 변환
                    drowsy_weight[2] = detector.calculate_eye_closing_time(ear)
                    # print(drowsy_weight[2])
                    # ----------------------------------------------------------------------------------------
                    # 눈 깜박임 분당 빈도수 - 1분마다 업데이트
                    drowsy_weight[3] = detector.calculate_blink_count_and_rate()
                    # print(drowsy_weight[3])
                    # ----------------------------------------------------------------------------------------
                    # perclos
                    # 눈 상태 업데이트
                    detector.update_eye_closure(ear)  # 현재 EAR 값과 현재 시간 전달
                    # 1분마다 업데이트
                    drowsy_weight[4] = detector.calculate_perclos()
                    # print(drowsy_weight[4])
                    # ----------------------------------------------------------------------------------------
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
            total_weight = sum(drowsy_weight)
            # 1단계, 2단계, 3단계, 4단계
            weight_ranges = [1.25, 2.50, 3.75, 5.00]

            if total_weight < weight_ranges[0]:
                drowy_level = 1
            elif total_weight < weight_ranges[1]:
                drowy_level = 2
            elif total_weight < weight_ranges[2]:
                drowy_level = 3
            elif total_weight >= weight_ranges[2]:
                drowy_level = 4

            # 여기에 drowy_level을 주기적으로 출력하는 코드를 추가
            current_time = time.time()
            if current_time - last_print_time >= 1:
                print(f"\nDrowsy Level: {drowy_level}, co2_level: {co2_level}")
                print(f"Drowsy Weight: {drowsy_weight}")
                print("Received:", received_int)
                last_print_time = current_time  # 마지막 출력 시간 업데이트

                # drowsy_weight 상태 업데이트
            text_status.update_status(drowsy_weight)

                # 이미지를 좌우 반전합니다.
            flipped_image = cv2.flip(image, 1)
                # co2
            # 텍스트와 배경색 추가
            add_text_with_background(flipped_image, text_status.received_int, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                                     (0, 255, 0), 1, (0, 0, 0))
            add_text_with_background(flipped_image, text_status.co2_status, (10, 90), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                                     (0, 0, 255), 1, (0, 0, 0))
            add_text_with_background(flipped_image, text_status.drowsy0_status, (10, 130), cv2.FONT_HERSHEY_COMPLEX,
                                     0.75, (0, 255, 255), 1, (0, 0, 0))
            add_text_with_background(flipped_image, text_status.drowsy1_status, (10, 160), cv2.FONT_HERSHEY_COMPLEX,
                                     0.75, (0, 255, 255), 1, (0, 0, 0))
            add_text_with_background(flipped_image, text_status.drowsy2_status, (10, 190), cv2.FONT_HERSHEY_COMPLEX,
                                     0.75, (0, 255, 255), 1, (0, 0, 0))
            add_text_with_background(flipped_image, text_status.drowsy3_status, (10, 220), cv2.FONT_HERSHEY_COMPLEX,
                                     0.75, (0, 255, 255), 1, (0, 0, 0))
            add_text_with_background(flipped_image, text_status.drowsy4_status, (10, 270), cv2.FONT_HERSHEY_COMPLEX,
                                     0.75, (0, 0, 255), 1, (0, 0, 0))

            cv2.imshow('MediaPipe Face Mesh', flipped_image)
            # 'Esc' 키 입력 시 루프 종료
            if cv2.waitKey(5) & 0xFF == 27:
                break

    picam2.stop()
    cv2.destroyAllWindows()


def main():

    # 신뢰할 수 있는 mac주소 연결?
    #client_socket, user_interest = setup_bluetooth_connection()  # 블루투스 연결 대기
    # 와이파이 연결
    thread = threading.Thread(target=serial_read)
    thread2 = threading.Thread(target=sensor.run)
    #thread3 = threading.Thread(target=charvis.run, args=(client_socket, user_interest))
    thread.daemon = True
    thread2.daemon = True
    #thread3.daemon = True
    thread.start()
    thread2.start()
    #thread3.start()
     #시리얼 데이터 수신 스레드 시작
    while True:
        drowy_run()
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

# 메인 함수 실행
if __name__ == "__main__":
    main()
