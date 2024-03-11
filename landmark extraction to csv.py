import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

######
mp_face_mesh = mp.solutions.face_mesh

face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                         min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles
######

sample_img = cv2.imread('/Users/reshadsazid/Human-Analytics-Project/droopy/droopy face 2.jpg')


# plt.figure(figsize=[10, 10])
# plt.title("Sample Image")
# plt.axis('off')
# plt.imshow(sample_img[:, :, ::-1])
# plt.show()


#####
def get_landmarks(image):
    face_mesh_results = face_mesh_images.process(sample_img[:, :, ::-1])
    LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
    RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
    LIPS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
    LEFT_EYE_BROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
    RIGHT_EYE_BROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))
    NOSE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_NOSE)))
    COUNTORS_INDEX = list(set(itertools.chain(*mp_face_mesh.FACEMESH_CONTOURS))) # wrong
    # RIGHT_CHEEK_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_CHEEK)))
    # MIDWAY_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_MIDWAY_BETWEEN_EYES)))
    """
    face_indexes = {
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
        "rightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],
        "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
        "rightEyebrowLower": [35, 124, 46, 53, 52, 65],
        "rightEyeIris": [473, 474, 475, 476, 477],
        "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
        "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
        "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
        "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
        "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
        "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
        "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],
        "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
        "leftEyebrowLower": [265, 353, 276, 283, 282, 295],
        "leftEyeIris": [468, 469, 470, 471, 472],
        "midwayBetweenEyes": [168],
        "noseTip": [1],
        "noseBottom": [2],
        "noseRightCorner": [98],
        "noseLeftCorner": [327],
        "rightCheek": [205],
        "leftCheek": [425]
    }
    
    """
    # The keys are already in string format, so no conversion is needed.
    # If you need any other modifications or have a different request, feel free to let me know!

    if face_mesh_results.multi_face_landmarks:

        for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):

            print(f'FACE NUMBER: {face_no + 1}')
            print('-----------------------')

            print(f'LEFT EYE LANDMARKS:\n')

            for LEFT_EYE_INDEX in LEFT_EYE_INDEXES:
                print(f"{LEFT_EYE_INDEX} : {face_landmarks.landmark[LEFT_EYE_INDEX]}")

            print(f'RIGHT EYE LANDMARKS:\n')

            for RIGHT_EYE_INDEX in RIGHT_EYE_INDEXES:
                print(f"{RIGHT_EYE_INDEX} : {face_landmarks.landmark[RIGHT_EYE_INDEX]}")

            print("LIPS LANDMARKS\n")

            for LIPS_INDEX in LIPS_INDEXES:
                print(f"{LIPS_INDEX} : {face_landmarks.landmark[LIPS_INDEX]}")

            print("LEFT EYEBROW LANDMARKS\n")
            for LEFT_EYE_BROW_INDEX in LEFT_EYE_BROW_INDEXES:
                print(f"{LEFT_EYE_BROW_INDEX} : {face_landmarks.landmark[LEFT_EYE_BROW_INDEX]}")
            print("RIGHT EYEBROW LANDMARKS\n")
            for RIGHT_EYE_BROW_INDEX in RIGHT_EYE_BROW_INDEXES:
                print(f"{RIGHT_EYE_BROW_INDEX} : {face_landmarks.landmark[RIGHT_EYE_BROW_INDEX]}")
            print("NOSE LANDMARKS\n")
            for NOSE_INDEX in NOSE_INDEXES:
                print(f"{NOSE_INDEX} : {face_landmarks.landmark[NOSE_INDEX]}")
            print("COUNTORS LANDMARKS\n")
            for COUNTORS_INDEX in COUNTORS_INDEX:
                print(f"{COUNTORS_INDEX} : {face_landmarks.landmark[COUNTORS_INDEX]}")
            # print("LEFT CHEEK LANDMARKS\n")
            # for LEFT_CHEEK_INDEX in LEFT_CHEEK_INDEXES:
            #     print(f"{LEFT_CHEEK_INDEX} : {face_landmarks.landmark[LEFT_CHEEK_INDEX]}")
            # print("RIGHT CHEEK LANDMARKS\n")
            # for RIGHT_EYE_INDEX in RIGHT_EYE_INDEXES:
            #     print(f"{RIGHT_EYE_INDEX} : {face_landmarks.landmark[RIGHT_EYE_INDEX]}")

get_landmarks(sample_img)

