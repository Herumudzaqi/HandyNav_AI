from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import time
import pyautogui
import threading
import screen_brightness_control as sbc
import HandTrackingModule as htm

app = Flask(__name__)

wCam, hCam = 640, 480
frameR = 20
wScr, hScr = pyautogui.size()

detector = htm.handDetector(maxHands=1, smooth_factor=0.0)
recording = False
recorder_thread = None
prev_brightness = 100  # nilai awal brightness global

def screen_record():
    global recording
    out = cv2.VideoWriter(f'screen_record_{int(time.time())}.avi',
                          cv2.VideoWriter_fourcc(*'XVID'), 15, (wScr, hScr))
    while recording:
        img = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(frame)
        time.sleep(0.05)
    out.release()

def handle_gestures(fingers, lmList, img):
    global recording, recorder_thread, prev_brightness

    x1, y1 = lmList[8][1:]  # jari telunjuk
    x_scr = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
    y_scr = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

    # Mode: None
    if fingers == [0, 0, 0, 0, 0]:
        return img

    # Klik (0 1 1 0 0)
    elif fingers == [0, 1, 1, 0, 0]:
        length, _, _ = detector.findDistance(8, 12, img, draw=False)
        if length < 40:
            pyautogui.click()
            time.sleep(0.1)

    # Mulai Rekam (0 1 1 1 1)
    elif fingers == [0, 1, 1, 1, 1] and not recording:
        recording = True
        recorder_thread = threading.Thread(target=screen_record, daemon=True)
        recorder_thread.start()

    # Stop Rekam (0 1 1 1 0)
    elif fingers == [0, 1, 1, 1, 0] and recording:
        recording = False
        if recorder_thread and recorder_thread.is_alive():
            recorder_thread.join()

    # Kontrol Brightness (1 1 0 0 0)
    elif fingers == [1, 1, 0, 0, 0]:
        try:
            length, _, _ = detector.findDistance(4, 8, img, draw=False)

            if length < 25:
                sbc.set_brightness(1)
                prev_brightness = 1
                cv2.putText(img, 'Brightness: MIN', (20, hCam - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                target_brightness = int(np.interp(y1, [frameR, hCam - frameR], [100, 1]))
                smooth_brightness = int(prev_brightness + (target_brightness - prev_brightness) * 0.2)
                sbc.set_brightness(smooth_brightness)
                prev_brightness = smooth_brightness
                cv2.putText(img, f'Brightness: {smooth_brightness}%', (20, hCam - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print("Brightness error:", e)

    # Gerak Kursor (1 1 1 1 1)
    if fingers == [1, 1, 1, 1, 1]:
        pyautogui.moveTo(x_scr, y_scr)

    return img

def generate_frames(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    cap.set(3, wCam)
    cap.set(4, hCam)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        img = detector.findHands(img, draw=True)
        lmList, _ = detector.findPosition(img, draw=False)

        if lmList:
            fingers = detector.fingersUp()
            img = handle_gestures(fingers, lmList, img)

        if recording:
            cv2.circle(img, (wCam - 40, 40), 10, (0, 0, 255), -1)
            cv2.putText(img, 'RECORDING', (wCam - 150, 47),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(img, f'FPS: {int(cap.get(cv2.CAP_PROP_FPS))}', (20, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    cam_id = request.args.get('cam', default=0, type=int)
    return Response(generate_frames(camera_index=cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
