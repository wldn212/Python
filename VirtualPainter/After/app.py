from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import HandTrackingModel as htm

app = Flask(__name__)

# Initialize global variables
brushThickness = 25
eraserThickness = 100
drawColor = (255, 0, 255)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Initialize detectors
detector = htm.HandDetector(detectionCon=0.65, maxHands=1)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

def generate_frames():
    global xp, yp, imgCanvas, drawColor
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        
        # Eye Tracking
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = img.shape
        
        # Hand Tracking
        img = detector.findHands(img)
        lmList, _ = detector.findPosition(img, draw=False)
        
        if len(lmList) >= 9:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            
            fingers = detector.fingersUp()
            
            # Selection Mode
            if fingers[1] and fingers[2]:
                if y1 < 125:
                    if 250 < x1 < 450:
                        drawColor = (255, 0, 255)  # Purple
                    elif 550 < x1 < 750:
                        drawColor = (255, 0, 0)  # Blue
                    elif 800 < x1 < 950:
                        drawColor = (0, 255, 0)  # Green
                    elif 1050 < x1 < 1200:
                        drawColor = (0, 0, 0)  # Eraser
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            
            # Drawing Mode
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                
                thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
                cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                xp, yp = x1, y1
            
            # Clear Canvas
            if all(x >= 1 for x in fingers):
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)
        
        # Merge canvas and camera feed
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        img = cv2.addWeighted(img, 0.7, imgCanvas, 0.3, 0)
        
        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)