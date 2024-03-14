"""
import os
from pathlib import Path

# Define the folder path
folder_path = 'path/to/your/folder'

# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    # Create the full file path
    old_file_path = os.path.join(folder_path, file_name)

    # Check if it's a file and not a directory
    if os.path.isfile(old_file_path):
        # Create a new file name with a prefix
        new_file_name = "new_" + file_name
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{file_name}' to '{new_file_name}'")

"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Your FACE_INDEXES dictionary remains the same
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

def draw_custom_landmarks(image, face_landmarks, face_indexes):
    img_height, img_width, _ = image.shape
    for group_name, indexes in face_indexes.items():
        for index in indexes:
            landmark = face_landmarks.landmark[index]
            # Convert normalized position to pixel position
            x = min(int(landmark.x * img_width), img_width - 1)
            y = min(int(landmark.y * img_height), img_height - 1)
            # Draw a small circle at the landmark position
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def get_and_draw_landmarks(image):
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            min_detection_confidence=0.5) as face_mesh:

        # Process the image and find face landmarks
        face_mesh_results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Check if any faces are detected
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                # Draw custom landmarks based on FACE_INDEXES
                draw_custom_landmarks(image, face_landmarks, FACE_INDEXES)

    return image


# Example usage
if __name__ == "__main__":
    # Load a sample image
    sample_img_path = '/Users/reshadsazid/Human-Analytics-Project/droopy/Stroke_Face_166.png'  # Update this path
    sample_img = cv2.imread(sample_img_path)

    if sample_img is not None:
        # Draw landmarks on the image
        annotated_image = get_and_draw_landmarks(sample_img.copy())

        # Display the annotated image
        cv2.imshow('Annotated Image', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Image could not be loaded.")
