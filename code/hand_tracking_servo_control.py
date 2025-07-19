import cv2
import mediapipe as mp
import time
import serial
import math

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_drawing = mp.solutions.drawing_utils

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 串口初始化
ser = serial.Serial('COM5', 9600)
time.sleep(2) 

# 平滑设置
previous_landmarks = None
alpha = 0.6

# 发送时间间隔
last_sent_time = 0
send_interval = 0.2

# 绘图函数
def draw_smoothed_landmarks(image, landmarks, connections):
    h, w, _ = image.shape
    points = []
    for lm in landmarks:
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        points.append((cx, cy))
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
    for connection in connections:
        start_idx, end_idx = connection
        cv2.line(image, points[start_idx], points[end_idx], (0, 255, 255), 2)

# 夹角计算
def calc_angle(p1, p2, p3):
    a = [p1[0]-p2[0], p1[1]-p2[1]]
    b = [p3[0]-p2[0], p3[1]-p2[1]]
    dot = a[0]*b[0] + a[1]*b[1]
    mag_a = math.hypot(a[0], a[1])
    mag_b = math.hypot(b[0], b[1])
    if mag_a * mag_b == 0:
        return 0
    angle_rad = math.acos(dot / (mag_a * mag_b))
    return math.degrees(angle_rad)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        current_landmarks = result.multi_hand_landmarks[0].landmark
        if previous_landmarks is None:
            smoothed_landmarks = [(lm.x, lm.y, lm.z) for lm in current_landmarks]
        else:
            smoothed_landmarks = []
            for i in range(21):
                x = alpha * current_landmarks[i].x + (1 - alpha) * previous_landmarks[i][0]
                y = alpha * current_landmarks[i].y + (1 - alpha) * previous_landmarks[i][1]
                z = alpha * current_landmarks[i].z + (1 - alpha) * previous_landmarks[i][2]
                smoothed_landmarks.append((x, y, z))
        previous_landmarks = smoothed_landmarks

        draw_smoothed_landmarks(frame, smoothed_landmarks, mp_hands.HAND_CONNECTIONS)

        # 食指三个关键点
        mcp = smoothed_landmarks[5]
        pip = smoothed_landmarks[6]
        dip = smoothed_landmarks[7]

        # 计算夹角
        angle = calc_angle(mcp, pip, dip)
        angle = max(90, min(180, angle))  # 限制在90~180之间
        servo_angle = int((180 - angle) / 90 * 160)  # 映射到 0°~160°
        print(f"夹角: {angle:.2f}° → 舵机: {servo_angle}°")

        # 每 0.2 秒发送一次
        current_time = time.time()
        if current_time - last_sent_time >= send_interval:
            ser.write((str(servo_angle) + '\n').encode())
            last_sent_time = current_time

    else:
        previous_landmarks = None

    cv2.imshow("手部识别", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
ser.close()
cv2.destroyAllWindows()

