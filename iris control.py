import cv2
import mediapipe as mp
import pyautogui

def main():
    # Initialize webcam and face mesh
    cam = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True)
    screen_w, screen_h = pyautogui.size()

    # Indices for the eyes and irises
    eye_indices = [
        33, 133, 7, 163, 144, 145, 153, 154, 155, 158, 159, 160, 161, 246, # Right eye
        362, 382, 263, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398  # Left eye
    ]
    iris_indices = [
        468, 469, 470, 471, 472, 473,  # Right iris
        474, 475, 476, 477, 478, 479   # Left iris
    ]

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            # Flip frame and convert to RGB
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = face_mesh.process(rgb_frame)

            if output.multi_face_landmarks:
                for face_landmarks in output.multi_face_landmarks:
                    eye_coords = collect_landmarks(face_landmarks, eye_indices + iris_indices, frame)
                    if eye_coords:
                        handle_mouse_movement(eye_coords, screen_w, screen_h)
                        draw_landmarks(frame, eye_coords)

            cv2.imshow('Eye Controlled Mouse', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        face_mesh.close()

def collect_landmarks(face_landmarks, indices, frame):
    eye_coords = []
    total_landmarks = len(face_landmarks.landmark)
    for index in indices:
        if index < total_landmarks:
            landmark = face_landmarks.landmark[index]
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            eye_coords.append((x, y))
    return eye_coords

def handle_mouse_movement(eye_coords, screen_w, screen_h):
    centroid_x = sum([coord[0] for coord in eye_coords]) / len(eye_coords)
    centroid_y = sum([coord[1] for coord in eye_coords]) / len(eye_coords)
    screen_x = screen_w * (centroid_x / screen_w)
    screen_y = screen_h * (centroid_y / screen_h)
    pyautogui.moveTo(screen_x, screen_y)

def draw_landmarks(frame, eye_coords):
    for x, y in eye_coords:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

if __name__ == "__main__":
    main()
