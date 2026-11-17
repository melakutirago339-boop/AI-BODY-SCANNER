
from flask import Flask, render_template, jsonify, request
import mediapipe as mp
import cv2
import numpy as np
import base64
import math
import logging

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)

mp_pose = mp.solutions.pose


# ---------------------------------
# BASIC ROUTES
# ---------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "application": "AI Body Scanner",
        "version": "2.0",
        "features": [
            "Automatic Body Detection",
            "Posture Analysis",
            "Body Measurement",
            "Health Screening Report",
            "33 Pose Landmarks"
        ]
    })


# ---------------------------------
# HELPER FUNCTIONS
# ---------------------------------

def get_distance(a, b, width, height):
    ax, ay = a.x * width, a.y * height
    bx, by = b.x * width, b.y * height

    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def get_midpoint(a, b):
    return {
        "x": (a.x + b.x) / 2,
        "y": (a.y + b.y) / 2
    }


def get_visibility(point):
    return float(point.visibility)


def calculate_tilt(a, b):
    return math.degrees(
        math.atan2(b.y - a.y, b.x - a.x)
    )


def visible(point, threshold=0.5):
    return get_visibility(point) >= threshold


# ---------------------------------
# POSTURE ANALYSIS
# ---------------------------------

def analyze_posture(points, view):

    left_shoulder = points[11]
    right_shoulder = points[12]

    left_hip = points[23]
    right_hip = points[24]

    left_knee = points[25]
    right_knee = points[26]

    left_ankle = points[27]
    right_ankle = points[28]

    shoulder_tilt = round(
        calculate_tilt(left_shoulder, right_shoulder), 2
    )

    hip_tilt = round(
        calculate_tilt(left_hip, right_hip), 2
    )

    shoulder_center = get_midpoint(
        left_shoulder, right_shoulder
    )

    hip_center = get_midpoint(
        left_hip, right_hip
    )

    vertical_difference = (
        shoulder_center["y"] - hip_center["y"]
    )

    horizontal_difference = (
        shoulder_center["x"] - hip_center["x"]
    )

    trunk_angle = round(
        math.degrees(
            math.atan2(
                abs(horizontal_difference),
                abs(vertical_difference) + 0.0001
            )
        ), 2
    )

    findings = []

    if view == "front" or view == "back":

        if abs(shoulder_tilt) > 5:
            findings.append(
                "Shoulder alignment difference observed"
            )

        if abs(hip_tilt) > 5:
            findings.append(
                "Hip alignment difference observed"
            )

    if view == "side":

        if trunk_angle > 12:
            findings.append(
                "Possible forward or backward trunk lean"
            )

    if not findings:
        findings.append(
            "No major alignment difference detected in this image"
        )

    return {
        "view": view,
        "shoulder_tilt_degrees": shoulder_tilt,
        "hip_tilt_degrees": hip_tilt,
        "trunk_angle_degrees": trunk_angle,
        "observations": findings,
        "status": "Visual posture analysis completed"
    }


# ---------------------------------
# BODY MEASUREMENTS
# ---------------------------------

def analyze_measurements(points, width, height, known_height_cm):

    left_shoulder = points[11]
    right_shoulder = points[12]

    left_hip = points[23]
    right_hip = points[24]

    left_ankle = points[27]
    right_ankle = points[28]

    shoulder_pixels = get_distance(
        left_shoulder,
        right_shoulder,
        width,
        height
    )

    hip_pixels = get_distance(
        left_hip,
        right_hip,
        width,
        height
    )

    ankle_center_y = (
        left_ankle.y + right_ankle.y
    ) / 2 * height

    nose_y = points[0].y * height

    visible_height_pixels = abs(
        ankle_center_y - nose_y
    )

    result = {
        "height": "Needs calibration",
        "shoulder_width": "Needs calibration",
        "hip_width": "Needs calibration",
        "proportions": "Pixel measurements available",
        "calibrated": False
    }

    if known_height_cm is not None:

        try:
            known_height_cm = float(known_height_cm)

            if not 80 <= known_height_cm <= 250:
                return result

            if visible_height_pixels <= 0:
                return result

            pixels_per_cm = (
                visible_height_pixels / known_height_cm
            )

            estimated_shoulder = (
                shoulder_pixels / pixels_per_cm
            )

            estimated_hip = (
                hip_pixels / pixels_per_cm
            )

            result = {
                "height": round(known_height_cm, 1),
                "shoulder_width": round(
                    estimated_shoulder, 1
                ),
                "hip_width": round(
                    estimated_hip, 1
                ),
                "proportions": {
                    "shoulder_to_hip_ratio": round(
                        estimated_shoulder /
                        max(estimated_hip, 0.1), 2
                    )
                },
                "calibrated": True,
                "measurement_note": (
                    "Approximate image-based estimates. "
                    "Accuracy depends on full-body visibility, "
                    "camera angle and calibration."
                )
            }

    return result


# ---------------------------------
# HEALTH SCREENING REPORT
# ---------------------------------

def create_screening_report(posture, measurements):

    return {
        "report_type": "Visual Body Posture Screening",
        "status": "Completed",
        "observations": posture["observations"],
        "measurements_available": measurements["calibrated"],
        "medical_diagnosis": False,
        "recommendation": (
            "This report is not a medical diagnosis. "
            "Consult a qualified healthcare professional "
            "for health concerns."
        )
    }


# ---------------------------------
# MAIN AI SCAN API
# ---------------------------------

@app.route("/api/scan", methods=["POST"])
def scan():

    try:

        data = request.get_json(silent=True)

        if not data or not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "Invalid request data"
            }), 400

        image_data = data.get("image")

        if not image_data or not isinstance(image_data, str):
            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(
                image_data,
                validate=True
            )
        except Exception:
            return jsonify({
                "success": False,
                "error": "Invalid image encoding"
            }), 400

        if len(image_bytes) > 8 * 1024 * 1024:
            return jsonify({
                "success": False,
                "error": "Image is too large"
            }), 413

        frame = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400

        height, width = frame.shape[:2]

        if width < 64 or height < 64:
            return jsonify({
                "success": False,
                "error": "Image resolution is too small"
            }), 400

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        view = str(
            data.get("view", "front")
        ).lower()

        if view not in ["front", "side", "back"]:
            view = "front"

        known_height_cm = data.get(
            "calibration_height_cm"
        )

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        ) as pose:

            results = pose.process(rgb)

        if not results.pose_landmarks:

            return jsonify({
                "success": True,
                "detected": False,
                "full_body_detected": False,
                "landmarks": 0,
                "posture": "Body not detected",
                "message": (
                    "Please stand in front of the camera "
                    "with your full body visible."
                )
            })

        points = results.pose_landmarks.landmark

        landmarks = []

        for index, point in enumerate(points):

            landmarks.append({
                "id": index,
                "x": round(float(point.x), 4),
                "y": round(float(point.y), 4),
                "z": round(float(point.z), 4),
                "visibility": round(
                    float(point.visibility), 4
                )
            })

        # Check important body parts

        important_parts = [
            11, 12, 23, 24, 25, 26, 27, 28
        ]

        detected_parts = sum(
            1 for i in important_parts
            if visible(points[i])
        )

        full_body_detected = (
            detected_parts >= 7
            and visible(points[27])
            and visible(points[28])
        )

        # Keep old API response compatible

        if not full_body_detected:

            return jsonify({
                "success": True,
                "detected": True,
                "full_body_detected": False,
                "landmarks": len(landmarks),
                "detected_parts": detected_parts,
                "posture": "Partial body detected",
                "message": (
                    "Move farther from the camera "
                    "and make sure your feet and head are visible."
                ),
                "points": landmarks
            })

        # Analyze posture

        posture_result = analyze_posture(
            points, view
        )

        # Analyze measurements

        measurements = analyze_measurements(
            points,
            width,
            height,
            known_height_cm
        )

        # Create screening report

        screening_report = create_screening_report(
            posture_result,
            measurements
        )

        # Final response

        return jsonify({

            "success": True,

            "detected": True,

            "full_body_detected": True,

            "landmarks": len(landmarks),

            "detected_parts": detected_parts,

            "posture": "Pose detected",

            "view": view,

            "height": measurements["height"],

            "shoulder": measurements["shoulder_width"],

            "hip_width": measurements["hip_width"],

            "proportions": measurements["proportions"],

            "measurements": measurements,

            "posture_analysis": posture_result,

            "health_screening": screening_report,

            "points": landmarks,

            "image": {
                "width": width,
                "height": height
            },

            "message": "AI Body Scan completed successfully"

        })

    except Exception:

        app.logger.exception("AI Body Scan failed")

        return jsonify({
            "success": False,
            "error": (
                "Server scan failed. "
                "Please check Render logs."
            )
        }), 500


# ---------------------------------
# ERROR HANDLERS
# ---------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error": "Uploaded request is too large"
    }), 413


@app.route("/api/status")
def api_status():

    return jsonify({
        "status": "ready",
        "scanner": "AI Body Scanner",
        "pose_detection": "MediaPipe",
        "version": "2.0"
    })


# ---------------------------------
# START SERVER
# ---------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
