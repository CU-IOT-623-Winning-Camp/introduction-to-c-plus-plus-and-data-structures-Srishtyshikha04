import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
points = []

def eye_brow_distance(landmarks, left_indices, right_indices):
    leye = np.array([landmarks[left_indices[0]].x, landmarks[left_indices[0]].y])
    reye = np.array([landmarks[right_indices[0]].x, landmarks[right_indices[0]].y])
    distq = np.linalg.norm(leye - reye)
    points.append(distq)
    return distq

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Example indices for eyebrows
            left_eyebrow_indices = [55, 65]
            right_eyebrow_indices = [285, 295]
            
            distq = eye_brow_distance(face_landmarks.landmark, left_eyebrow_indices, right_eyebrow_indices)
            stress_value = np.exp(-abs(distq - np.min(points)) / abs(np.max(points) - np.min(points)))
            
            label = "High Stress" if stress_value >= 0.75 else "Low Stress"
            cv2.putText(frame, f"Stress Level: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
