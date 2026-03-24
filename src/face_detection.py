"""
Face detection module for verification
"""

import cv2
import numpy as np
from pathlib import Path


# Load pre-trained face cascade classifier
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def detect_face_in_frame(frame: np.ndarray) -> tuple[bool, np.ndarray]:
    """
    Detect face in a frame using Haar Cascade.
    Returns (face_detected: bool, annotated_frame: np.ndarray)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    face_detected = len(faces) > 0
    
    # Draw rectangles around detected faces
    annotated_frame = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return face_detected, annotated_frame


def simple_liveness_check(frames: list[np.ndarray]) -> bool:
    """
    Simple liveness check: detect if there's movement between frames.
    Returns True if liveness detected (movement between frames).
    """
    if len(frames) < 2:
        return False
    
    # Compare consecutive frames for movement
    flow_detected = False
    for i in range(len(frames) - 1):
        frame1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(frame1, frame2)
        non_zero = cv2.countNonZero(diff)
        
        if non_zero > 500:  # Threshold for movement
            flow_detected = True
            break
    
    return flow_detected


def verify_selfie_webcam(timeout_seconds: int = 30) -> dict:
    """
    Capture selfie from webcam and verify face detection.
    Returns: {
        'verified': bool,
        'face_detected': bool,
        'liveness_detected': bool,
        'message': str,
        'frame': arr or None
    }
    """
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {
                "verified": False,
                "face_detected": False,
                "message": "Camera not available",
                "frame": None,
            }
        
        frames_captured = []
        start_time = cv2.getTickCount()
        face_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for selfie (mirror effect)
            frame = cv2.flip(frame, 1)
            
            # Detect face
            face_detected, annotated_frame = detect_face_in_frame(frame)
            
            if face_detected:
                face_count += 1
                frames_captured.append(frame)
            
            # Check timeout
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed > timeout_seconds:
                break
            
            # Display frame with instruction
            cv2.putText(
                annotated_frame,
                f"Face Detection: {'Detected' if face_detected else 'Waiting...'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if face_detected else (0, 0, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Time: {elapsed:.1f}s",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            
            # Show frame in small window (commented for Streamlit compatibility)
            # cv2.imshow("Selfie Verification", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Determine verification result
        verified = face_count >= 5  # At least 5 frames with face detected
        liveness_detected = simple_liveness_check(frames_captured) if len(frames_captured) > 1 else False
        
        result_frame = frames_captured[-1] if frames_captured else None
        
        return {
            "verified": verified,
            "face_detected": face_count > 0,
            "liveness_detected": liveness_detected,
            "message": (
                "✅ Verification successful! Face detected and liveness confirmed."
                if verified else
                "❌ Verification failed! Please try again."
            ),
            "frame": result_frame,
        }
    
    except Exception as e:
        return {
            "verified": False,
            "face_detected": False,
            "message": f"Error: {str(e)}",
            "frame": None,
        }
