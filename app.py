

from flask import Flask, render_template, jsonify, request
import mediapipe as mp
import cv2
import numpy as np

app = Flask(__name__)

mp_pose = mp.solutions.pose

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "application": "AI Body Scanner"
    })

@app.route("/detect-pose", methods=["POST"])
def detect_pose():
    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400

        file = request.files["image"]

        image_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        ) as pose:

            results = pose.process(rgb_frame)

            if not results.pose_landmarks:
                return jsonify({
                    "success": True,
                    "detected": False,
                    "message": "No full body detected"
                })

            landmarks = []

            for point in results.pose_landmarks.landmark:
                landmarks.append({
                    "x": round(point.x, 4),
                    "y": round(point.y, 4),
                    "visibility": round(point.visibility, 4)
                })

        return jsonify({
            "success": True,
            "detected": True,
            "landmarks": landmarks
        })

    except Exception:
        app.logger.exception("Pose detection failed")
        return jsonify({
            "success": False,
            "error": "Pose detection failed on server"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
