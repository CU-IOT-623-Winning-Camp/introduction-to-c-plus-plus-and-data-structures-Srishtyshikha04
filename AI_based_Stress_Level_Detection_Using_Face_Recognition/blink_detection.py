from scipy.spatial import distance as dist
import numpy as np
import cv2
import mediapipe as mp

# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between vertical landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between horizontal landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    eye_opening_ratio = (A + B) / (2.0 * C)

    return eye_opening_ratio


# Threshold and consecutive frame count for blink detection
ar_thresh = 0.3
eye_ar_consec_frame = 5
counter = 0
total = 0

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for face landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract the eye landmarks (indices are based on Mediapipe's face mesh model)
            left_eye_indices = [362, 385, 387, 263, 373, 380]
            right_eye_indices = [33, 160, 158, 133, 153, 144]

            left_eye = np.array([[face_landmarks.landmark[i].x * frame.shape[1],
                                  face_landmarks.landmark[i].y * frame.shape[0]] for i in left_eye_indices])
            right_eye = np.array([[face_landmarks.landmark[i].x * frame.shape[1],
                                   face_landmarks.landmark[i].y * frame.shape[0]] for i in right_eye_indices])

            # Draw the eye contours
            cv2.polylines(frame, [np.int32(left_eye)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.int32(right_eye)], True, (0, 255, 0), 1)

            # Calculate the EAR
            left_eye_EAR = eye_aspect_ratio(left_eye)
            right_eye_EAR = eye_aspect_ratio(right_eye)

            avg_EAR = (left_eye_EAR + right_eye_EAR) / 2.0

            # Blink detection
            if avg_EAR < ar_thresh:
                counter += 1
            else:
                if counter > eye_ar_consec_frame:
                    total += 1
                counter = 0

            # Display blink count and EAR
            cv2.putText(frame, f"Blinks: {total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit loop when 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()
