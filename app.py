import cv2
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# Load model YOLOv8x (dilakukan sekali saat server start)
model = YOLO("yolov8n.pt")

# Global camera object
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Baca frame dari kamera
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize frame untuk stabilitas & kecocokan imgsz (opsional)
            # frame = cv2.resize(frame, (1280, 720))

            # Jalankan inferensi YOLOv8x
            # Parameter: imgsz 1280, conf 0.25, iou 0.5, hanya class person (id 0)
            results = model.predict(
                source=frame,
                imgsz=1280,
                conf=0.25,
                iou=0.5,
                classes=[0],
                verbose=False
            )

            # Hitung jumlah orang
            person_count = 0
            if len(results) > 0:
                res = results[0]
                person_count = len(res.boxes)

                # Gambar bounding box dan label ke frame
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Bounding box hijau
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Label person + confidence
                    label = f"person {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Tampilkan angka jumlah orang di pojok kiri atas frame
            cv2.putText(
                frame, 
                f"Jumlah Orang: {person_count}", 
                (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, 
                (0, 0, 255), # Merah agar terlihat jelas
                3
            )

            # Encode frame ke JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Format MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Halaman utama live feed."""
    return render_template('live.html')

@app.route('/video')
def video_feed():
    """Endpoint stream MJPEG."""
    if not camera.isOpened():
        return "Kamera tidak tersedia", 503
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Pastikan server jalan di host 0.0.0.0 agar bisa diakses device lain di jaringan yang sama
    app.run(host='0.0.0.0', port=5000, debug=False)
