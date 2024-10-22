import cv2
import mediapipe as mp
import numpy as np
import itertools
# Initialize face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
#sample_img = cv2.imread('/Users/reshadsazid/Human-Analytics-Project/droopy/droopy face 1.jpg')

#normalization added
def normalize_landmarks(landmarks):
    mean = np.mean(landmarks, axis=0)
    std = np.std(landmarks, axis=0)
    normalized_landmarks = (landmarks - mean) / std
    return normalized_landmarks

def get_landmarks(image):
    face_mesh_results = face_mesh_images.process(image[:, :, ::-1])
    # Define your FACE_INDEXES here
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
        "rightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],
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
        "leftCheek": [425]
    }

    data = []

    if face_mesh_results.multi_face_landmarks:
        for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            all_landmarks = []
            for landmarks_group, indexes in FACE_INDEXES.items():
                for index in indexes:
                    landmark = face_landmarks.landmark[index]
                    all_landmarks.append([landmark.x, landmark.y, landmark.z])

            all_landmarks_np = np.array(all_landmarks)
            normalized_landmarks_np = normalize_landmarks(all_landmarks_np)

            # Correctly append individual landmarks
            i = 0
            for landmarks_group, indexes in FACE_INDEXES.items():
                for index in indexes:
                    normalized_landmark = normalized_landmarks_np[i]
                    i += 1
                    # This line assumes every index is processed correctly
                    data.append((landmarks_group, index, *normalized_landmark))
    return data


"""
"""======================================================================"""
#adding normalization [the standardized score method]

import os

folder_path ='/Users/reshadsazid/Human-Analytics-Project/straight face'

all_landmarks = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".png")):
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path)
        landmarks = get_landmarks(image)
        for landmark in landmarks:
            # Add filename to distinguish between images
            all_landmarks.append((filename,) + landmark)



"""======================================================================"""

#export csv
import pandas as pd

# Convert the list of tuples into a DataFrame
df = pd.DataFrame(all_landmarks, columns=['Filename', 'Landmark Group', 'Index', 'X', 'Y', 'Z'])

# Write the DataFrame to a CSV file
csv_output_path = '/Users/reshadsazid/Human-Analytics-Project/straight face landmarks.csv'
df.to_csv(csv_output_path, index=False)


"""==========================IMPLEMENTING FLATTENED LANDMARKS=========================="""

import pandas as pd

# Load the CSV into a DataFrame
df = pd.read_csv('droopy landmarks.csv')

# Generate a unique identifier for each landmark
df['unique_id'] = df['Filename'] + '_' + df['Landmark Group'] + '_' + df['Index'].astype(str)

# Create wide-format DataFrames for each coordinate
x_wide = df.pivot(index='Filename', columns='unique_id', values='X')
y_wide = df.pivot(index='Filename', columns='unique_id', values='Y')
z_wide = df.pivot(index='Filename', columns='unique_id', values='Z')

# Concatenate the wide-format DataFrames
flattened_df = pd.concat([x_wide, y_wide.add_suffix('_Y'), z_wide.add_suffix('_Z')], axis=1).reset_index()

flattened_df.to_csv('flattened_straight face_landmarks.csv', index=False)

"""