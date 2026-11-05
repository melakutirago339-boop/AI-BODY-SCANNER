
from flask import Flask, render_template, jsonify, request
import mediapipe as mp
import cv2
import numpy as np
import base64

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "application": "AI Body Scanner"
    })

@app.route("/api/scan", methods=["POST"])
def scan():
    try:
        data = request.get_json(silent=True)

        if not data or "image" not in data:
            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400

        image_data = data["image"]

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)

        frame = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

       results = pose_detector.process(rgb)

if not results.pose_landmarks:
    return jsonify({
        "success": True,
        "detected": False,
        "landmarks": 0,
        "points": [],
        "posture": "Full body not detected"
    })

points = results.pose_landmarks.landmark

            landmarks = []

            for point in points:
                landmarks.append({
                    "x": round(point.x, 4),
                    "y": round(point.y, 4),
                    "visibility": round(point.visibility, 4)
                })

        return jsonify({
            "success": True,
            "detected": True,
            "landmarks": len(landmarks),
            "posture": "Pose detected",
            "height": "Needs calibration",
            "shoulder": "Needs calibration",
            "proportions": "Pose landmarks available",
            "points": landmarks
        })

    except Exception:
        app.logger.exception("Scan failed")
        return jsonify({
            "success": False,
            "error": "Server scan failed. Check Render logs."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
