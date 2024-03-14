import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")



mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
sample_img = cv2.imread('/Users/reshadsazid/Human-Analytics-Project/droopy/droopy face 1.jpg')


def get_landmarks(image):
    face_mesh_results = face_mesh_images.process(sample_img[:, :, ::-1])
    FACE_INDEXES = {
        "silhouette": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                       397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                       172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
        "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
        "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],

        "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
        "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
        "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
        "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
        "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
        "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
        r"ightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],

        "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
        "rightEyebrowLower": [35, 124, 46, 53, 52, 65],

        "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
        "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
        "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
        "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
        "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
        "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
        "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],

        "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
        "leftEyebrowLower": [265, 353, 276, 283, 282, 295],

        "midwayBetweenEyes": [168],

        "noseTip": [1],
        "noseBottom": [2],
        "noseRightCorner": [98],
        "noseLeftCorner": [327],

        "rightCheek": [205],
        "leftCheek": [425]}
    if face_mesh_results.multi_face_landmarks:
        c = 0
        for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            for landmarks, indexes in FACE_INDEXES.items():
                print(landmarks)
                for index in indexes:
                    print(f"{index} : {face_landmarks.landmark[index]}")
                    c += 1
        print(c)


get_landmarks(sample_img)




"============================================="

import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
FACE_INDEXES = {
        "silhouette": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                       397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                       172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
        "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
        "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],

        "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
        "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
        "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
        "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
        "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
        "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
        r"ightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],

        "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
        "rightEyebrowLower": [35, 124, 46, 53, 52, 65],

        "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
        "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
        "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
        "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
        "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
        "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
        "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],

        "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
        "leftEyebrowLower": [265, 353, 276, 283, 282, 295],

        "midwayBetweenEyes": [168],

        "noseTip": [1],
        "noseBottom": [2],
        "noseRightCorner": [98],
        "noseLeftCorner": [327],

        "rightCheek": [205],
        "leftCheek": [425]}

mp_drawing = mp.solutions.drawing_utils
my_drawing_specs = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

# Load your image
image_path = '/Users/reshadsazid/Human-Analytics-Project/droopy/droopy face 1.jpg'
image = cv2.imread(image_path)

# Convert the BGR image to RGB before processing
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process with FaceMesh
face_results = face_mesh.process(image_rgb)

# Draw specific FaceMesh landmarks using FACE_INDEXES
if face_results.multi_face_landmarks:
    for face_landmarks in face_results.multi_face_landmarks:
        for region, indexes in FACE_INDEXES.items():
            for index in indexes:
                # Convert normalized position to pixel values
                x = int(face_landmarks.landmark[index].x * image.shape[1])
                y = int(face_landmarks.landmark[index].y * image.shape[0])
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# Display the image
cv2.imshow('Facial Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the image to a file
#cv2.imwrite('/path/to/save/image.jpg', image)
