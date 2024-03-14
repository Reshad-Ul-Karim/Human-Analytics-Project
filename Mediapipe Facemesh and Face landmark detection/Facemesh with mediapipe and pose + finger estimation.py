import cv2
import mediapipe as mp

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
my_drawing_specs = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

# Holistic and FaceMesh models
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

# Initialize Holistic and FaceMesh models
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
        mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB before processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with Holistic
        holistic_results = holistic.process(image_rgb)

        # Process with FaceMesh
        face_results = face_mesh.process(image_rgb)

        # Convert back to BGR for rendering
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw Holistic landmarks (body and hands)
        if holistic_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                holistic_results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )
        '''
        if holistic_results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                holistic_results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )
        '''
        if holistic_results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                holistic_results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
        '''
        if holistic_results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                holistic_results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )
        '''
        if holistic_results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                holistic_results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Draw FaceMesh landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=image_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=my_drawing_specs
                )

        cv2.imshow("My video capture", cv2.flip(image_bgr, 1))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
