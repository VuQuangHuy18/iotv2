from flask import Flask, render_template, Response
import cv2
from flask_socketio import SocketIO
import numpy as np
from ultralytics import YOLO
import time
import RPi.GPIO as GPIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
model = YOLO('best.pt')  # Đảm bảo cập nhật đường dẫn đến mô hình YOLO đã được đào tạo

# Cấu hình GPIO cho các servo
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)  # Servo 1
GPIO.setup(27, GPIO.OUT)  # Servo 2
GPIO.setup(22, GPIO.OUT)  # Servo 3

servo1 = GPIO.PWM(17, 50)
servo2 = GPIO.PWM(27, 50)
servo3 = GPIO.PWM(22, 50)

servo1.start(0)
servo2.start(0)
servo3.start(0)

# Cấu hình GPIO cho các cảm biến IR
GPIO.setup(23, GPIO.IN)  # Cảm biến IR 1
GPIO.setup(24, GPIO.IN)  # Cảm biến IR 2
GPIO.setup(25, GPIO.IN)  # Cảm biến IR 3

# Hàm điều khiển các servo và kiểm tra cảm biến sau khi servo đóng
def rotate_servo_and_check(servo, angle, sensor_pin):
    # Quay servo tới góc chỉ định
    duty = angle / 18 + 2
    GPIO.output(servo, True)
    servo.ChangeDutyCycle(duty)
    time.sleep(1)  # Chờ servo quay xong
    GPIO.output(servo, False)
    servo.ChangeDutyCycle(0)

    # Sau khi servo đóng lại, kiểm tra trạng thái cảm biến
    time.sleep(1)  # Chờ thêm 1 giây sau khi đóng
    if GPIO.input(sensor_pin) == GPIO.HIGH:
        print(f"Thùng rác gắn với cảm biến {sensor_pin} đầy.")
    else:
        print(f"Thùng rác gắn với cảm biến {sensor_pin} chưa đầy.")

# Hàm xử lý khung hình
def process_frame(frame):
    # Xác định vùng quan tâm (ROI)
    x1, y1, x2, y2 = 200, 0, 600, frame.shape[0]
    frame_cut = frame[y1:y2, x1:x2]

    # Vẽ hình chữ nhật ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Chạy YOLOv8 tracking trên ROI
    results = model.track(frame_cut, persist=True)

    objects_detected = set()

    for result in results:
        boxes = result.boxes.numpy()
        for box in boxes:
            class_idx = int(box.cls[0])
            class_name = model.names[class_idx]

            x11, y11 = int(box.xyxy[0][0]) + x1, int(box.xyxy[0][1]) + y1
            x21, y21 = int(box.xyxy[0][2]) + x1, int(box.xyxy[0][3]) + y1

            cv2.rectangle(frame, (x11, y11), (x21, y21), (255, 255, 0), 2)
            cv2.putText(frame, class_name, (x11, y11 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            objects_detected.add(class_name)

    # Kiểm tra vật thể phát hiện và điều khiển servo
    for class_name in objects_detected:
        if class_name == "plastic_bottle":
            rotate_servo_and_check(servo1, 180, 23)  # Quay servo 1 và kiểm tra cảm biến 1 (GPIO 23)
        elif class_name == "paper":
            rotate_servo_and_check(servo2, 180, 24)  # Quay servo 2 và kiểm tra cảm biến 2 (GPIO 24)
        elif class_name == "food_box":
            rotate_servo_and_check(servo3, 180, 25)  # Quay servo 3 và kiểm tra cảm biến 3 (GPIO 25)

    return frame  # Trả về frame đã xử lý

# Hàm tạo các khung hình cho video feed
def generate_frames():
    cap = cv2.VideoCapture("http://192.168.100.110:4747/video")
    retry_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            retry_count += 1
            print("Không nhận được khung hình từ camera. Đang thử lại...")
            if retry_count > 5:
                print("Camera không hoạt động. Thoát...")
                break
            time.sleep(1)
            continue

        retry_count = 0  # Reset nếu thành công
        frame = cv2.flip(frame, 1)  # Lật ngang
        frame = process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Hàm dọn dẹp GPIO khi thoát
def cleanup():
    servo1.stop()
    servo2.stop()
    servo3.stop()
    GPIO.cleanup()

if __name__ == '__main__':
    try:
        socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        cleanup()
