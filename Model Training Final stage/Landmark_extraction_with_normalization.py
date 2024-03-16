import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np
from collections import defaultdict

# Initialize face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def normalize_landmarks(landmarks):
    mean = np.mean(landmarks, axis=0)
    std = np.std(landmarks, axis=0)
    normalized_landmarks = (landmarks - mean) / std
    return normalized_landmarks


def get_landmarks(image):
    face_mesh_results = face_mesh_images.process(image[:, :, ::-1])
    # Define your FACE_INDEXES here
    FACE_INDEXES = {
        # "silhouette": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        #                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        #                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
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
        "rightCheek": [205, 137, 123, 50, 203, 177, 147, 187, 207, 216, 215, 213, 192, 214, 212, 138, 135, 210, 169],
        "leftCheek": [425, 352, 280, 330, 266, 423, 426, 427, 411, 376, 436, 416, 432, 434, 422, 430, 364, 394, 371]
    }

    data = []

    if face_mesh_results.multi_face_landmarks:
        for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            all_landmarks = []
            for landmarks_group, indexes in FACE_INDEXES.items():
                for index in indexes:
                    landmark = face_landmarks.landmark[index]
                    all_landmarks.append([landmark.x, landmark.y])

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


# folder_path = "Normal_Face_Dataset"
folder_path = "Final Data Set Normal"
all_landmarks = defaultdict(list)

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".png")):  # Check for both .jpg and .png files
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue
        landmarks = get_landmarks(image)
        for landmark in landmarks:
            group, index, x, y = landmark
            all_landmarks[filename].extend([x, y])
to_csv = []
for filename, landmarks in all_landmarks.items():
    out = [filename, *landmarks]
    if len(out) == 457:
        to_csv.append(out)
    else:
        print(filename, len(out))
# Convert the list of tuples into a DataFrame
df = pd.DataFrame(to_csv,
                  columns=['Filename', 'lipsUpperOuter_61_x', 'lipsUpperOuter_61_y', 'lipsUpperOuter_185_x',
                           'lipsUpperOuter_185_y',
                           'lipsUpperOuter_40_x', 'lipsUpperOuter_40_y', 'lipsUpperOuter_39_x', 'lipsUpperOuter_39_y',
                           'lipsUpperOuter_37_x', 'lipsUpperOuter_37_y', 'lipsUpperOuter_0_x', 'lipsUpperOuter_0_y',
                           'lipsUpperOuter_267_x', 'lipsUpperOuter_267_y', 'lipsUpperOuter_269_x',
                           'lipsUpperOuter_269_y', 'lipsUpperOuter_270_x', 'lipsUpperOuter_270_y',
                           'lipsUpperOuter_409_x', 'lipsUpperOuter_409_y', 'lipsUpperOuter_291_x',
                           'lipsUpperOuter_291_y', 'lipsLowerOuter_146_x', 'lipsLowerOuter_146_y',
                           'lipsLowerOuter_91_x', 'lipsLowerOuter_91_y', 'lipsLowerOuter_181_x', 'lipsLowerOuter_181_y',
                           'lipsLowerOuter_84_x', 'lipsLowerOuter_84_y', 'lipsLowerOuter_17_x', 'lipsLowerOuter_17_y',
                           'lipsLowerOuter_314_x', 'lipsLowerOuter_314_y', 'lipsLowerOuter_405_x',
                           'lipsLowerOuter_405_y', 'lipsLowerOuter_321_x', 'lipsLowerOuter_321_y',
                           'lipsLowerOuter_375_x', 'lipsLowerOuter_375_y', 'lipsLowerOuter_291_x',
                           'lipsLowerOuter_291_y', 'lipsUpperInner_78_x', 'lipsUpperInner_78_y', 'lipsUpperInner_191_x',
                           'lipsUpperInner_191_y', 'lipsUpperInner_80_x', 'lipsUpperInner_80_y', 'lipsUpperInner_81_x',
                           'lipsUpperInner_81_y', 'lipsUpperInner_82_x', 'lipsUpperInner_82_y', 'lipsUpperInner_13_x',
                           'lipsUpperInner_13_y', 'lipsUpperInner_312_x', 'lipsUpperInner_312_y',
                           'lipsUpperInner_311_x', 'lipsUpperInner_311_y', 'lipsUpperInner_310_x',
                           'lipsUpperInner_310_y', 'lipsUpperInner_415_x', 'lipsUpperInner_415_y',
                           'lipsUpperInner_308_x', 'lipsUpperInner_308_y', 'lipsLowerInner_78_x', 'lipsLowerInner_78_y',
                           'lipsLowerInner_95_x', 'lipsLowerInner_95_y', 'lipsLowerInner_88_x', 'lipsLowerInner_88_y',
                           'lipsLowerInner_178_x', 'lipsLowerInner_178_y', 'lipsLowerInner_87_x', 'lipsLowerInner_87_y',
                           'lipsLowerInner_14_x', 'lipsLowerInner_14_y', 'lipsLowerInner_317_x', 'lipsLowerInner_317_y',
                           'lipsLowerInner_402_x', 'lipsLowerInner_402_y', 'lipsLowerInner_318_x',
                           'lipsLowerInner_318_y', 'lipsLowerInner_324_x', 'lipsLowerInner_324_y',
                           'lipsLowerInner_308_x', 'lipsLowerInner_308_y', 'rightEyeUpper0_246_x',
                           'rightEyeUpper0_246_y', 'rightEyeUpper0_161_x', 'rightEyeUpper0_161_y',
                           'rightEyeUpper0_160_x', 'rightEyeUpper0_160_y', 'rightEyeUpper0_159_x',
                           'rightEyeUpper0_159_y', 'rightEyeUpper0_158_x', 'rightEyeUpper0_158_y',
                           'rightEyeUpper0_157_x', 'rightEyeUpper0_157_y', 'rightEyeUpper0_173_x',
                           'rightEyeUpper0_173_y', 'rightEyeLower0_33_x', 'rightEyeLower0_33_y', 'rightEyeLower0_7_x',
                           'rightEyeLower0_7_y', 'rightEyeLower0_163_x', 'rightEyeLower0_163_y', 'rightEyeLower0_144_x',
                           'rightEyeLower0_144_y', 'rightEyeLower0_145_x', 'rightEyeLower0_145_y',
                           'rightEyeLower0_153_x', 'rightEyeLower0_153_y', 'rightEyeLower0_154_x',
                           'rightEyeLower0_154_y', 'rightEyeLower0_155_x', 'rightEyeLower0_155_y',
                           'rightEyeLower0_133_x', 'rightEyeLower0_133_y', 'rightEyeUpper1_247_x',
                           'rightEyeUpper1_247_y', 'rightEyeUpper1_30_x', 'rightEyeUpper1_30_y', 'rightEyeUpper1_29_x',
                           'rightEyeUpper1_29_y', 'rightEyeUpper1_27_x', 'rightEyeUpper1_27_y', 'rightEyeUpper1_28_x',
                           'rightEyeUpper1_28_y', 'rightEyeUpper1_56_x', 'rightEyeUpper1_56_y', 'rightEyeUpper1_190_x',
                           'rightEyeUpper1_190_y', 'rightEyeLower1_130_x', 'rightEyeLower1_130_y',
                           'rightEyeLower1_25_x', 'rightEyeLower1_25_y', 'rightEyeLower1_110_x', 'rightEyeLower1_110_y',
                           'rightEyeLower1_24_x', 'rightEyeLower1_24_y', 'rightEyeLower1_23_x', 'rightEyeLower1_23_y',
                           'rightEyeLower1_22_x', 'rightEyeLower1_22_y', 'rightEyeLower1_26_x', 'rightEyeLower1_26_y',
                           'rightEyeLower1_112_x', 'rightEyeLower1_112_y', 'rightEyeLower1_243_x',
                           'rightEyeLower1_243_y', 'rightEyeUpper2_113_x', 'rightEyeUpper2_113_y',
                           'rightEyeUpper2_225_x', 'rightEyeUpper2_225_y', 'rightEyeUpper2_224_x',
                           'rightEyeUpper2_224_y', 'rightEyeUpper2_223_x', 'rightEyeUpper2_223_y',
                           'rightEyeUpper2_222_x', 'rightEyeUpper2_222_y', 'rightEyeUpper2_221_x',
                           'rightEyeUpper2_221_y', 'rightEyeUpper2_189_x', 'rightEyeUpper2_189_y',
                           'rightEyeLower2_226_x', 'rightEyeLower2_226_y', 'rightEyeLower2_31_x', 'rightEyeLower2_31_y',
                           'rightEyeLower2_228_x', 'rightEyeLower2_228_y', 'rightEyeLower2_229_x',
                           'rightEyeLower2_229_y', 'rightEyeLower2_230_x', 'rightEyeLower2_230_y',
                           'rightEyeLower2_231_x', 'rightEyeLower2_231_y', 'rightEyeLower2_232_x',
                           'rightEyeLower2_232_y', 'rightEyeLower2_233_x', 'rightEyeLower2_233_y',
                           'rightEyeLower2_244_x', 'rightEyeLower2_244_y', 'rightEyeLower3_143_x',
                           'rightEyeLower3_143_y', 'rightEyeLower3_111_x', 'rightEyeLower3_111_y',
                           'rightEyeLower3_117_x', 'rightEyeLower3_117_y', 'rightEyeLower3_118_x',
                           'rightEyeLower3_118_y', 'rightEyeLower3_119_x', 'rightEyeLower3_119_y',
                           'rightEyeLower3_120_x', 'rightEyeLower3_120_y', 'rightEyeLower3_121_x',
                           'rightEyeLower3_121_y', 'rightEyeLower3_128_x', 'rightEyeLower3_128_y',
                           'rightEyeLower3_245_x', 'rightEyeLower3_245_y', 'rightEyebrowUpper_156_x',
                           'rightEyebrowUpper_156_y', 'rightEyebrowUpper_70_x', 'rightEyebrowUpper_70_y',
                           'rightEyebrowUpper_63_x', 'rightEyebrowUpper_63_y', 'rightEyebrowUpper_105_x',
                           'rightEyebrowUpper_105_y', 'rightEyebrowUpper_66_x', 'rightEyebrowUpper_66_y',
                           'rightEyebrowUpper_107_x', 'rightEyebrowUpper_107_y', 'rightEyebrowUpper_55_x',
                           'rightEyebrowUpper_55_y', 'rightEyebrowUpper_193_x', 'rightEyebrowUpper_193_y',
                           'rightEyebrowLower_35_x', 'rightEyebrowLower_35_y', 'rightEyebrowLower_124_x',
                           'rightEyebrowLower_124_y', 'rightEyebrowLower_46_x', 'rightEyebrowLower_46_y',
                           'rightEyebrowLower_53_x', 'rightEyebrowLower_53_y', 'rightEyebrowLower_52_x',
                           'rightEyebrowLower_52_y', 'rightEyebrowLower_65_x', 'rightEyebrowLower_65_y',
                           'leftEyeUpper0_466_x', 'leftEyeUpper0_466_y', 'leftEyeUpper0_388_x', 'leftEyeUpper0_388_y',
                           'leftEyeUpper0_387_x', 'leftEyeUpper0_387_y', 'leftEyeUpper0_386_x', 'leftEyeUpper0_386_y',
                           'leftEyeUpper0_385_x', 'leftEyeUpper0_385_y', 'leftEyeUpper0_384_x', 'leftEyeUpper0_384_y',
                           'leftEyeUpper0_398_x', 'leftEyeUpper0_398_y', 'leftEyeLower0_263_x', 'leftEyeLower0_263_y',
                           'leftEyeLower0_249_x', 'leftEyeLower0_249_y', 'leftEyeLower0_390_x', 'leftEyeLower0_390_y',
                           'leftEyeLower0_373_x', 'leftEyeLower0_373_y', 'leftEyeLower0_374_x', 'leftEyeLower0_374_y',
                           'leftEyeLower0_380_x', 'leftEyeLower0_380_y', 'leftEyeLower0_381_x', 'leftEyeLower0_381_y',
                           'leftEyeLower0_382_x', 'leftEyeLower0_382_y', 'leftEyeLower0_362_x', 'leftEyeLower0_362_y',
                           'leftEyeUpper1_467_x', 'leftEyeUpper1_467_y', 'leftEyeUpper1_260_x', 'leftEyeUpper1_260_y',
                           'leftEyeUpper1_259_x', 'leftEyeUpper1_259_y', 'leftEyeUpper1_257_x', 'leftEyeUpper1_257_y',
                           'leftEyeUpper1_258_x', 'leftEyeUpper1_258_y', 'leftEyeUpper1_286_x', 'leftEyeUpper1_286_y',
                           'leftEyeUpper1_414_x', 'leftEyeUpper1_414_y', 'leftEyeLower1_359_x', 'leftEyeLower1_359_y',
                           'leftEyeLower1_255_x', 'leftEyeLower1_255_y', 'leftEyeLower1_339_x', 'leftEyeLower1_339_y',
                           'leftEyeLower1_254_x', 'leftEyeLower1_254_y', 'leftEyeLower1_253_x', 'leftEyeLower1_253_y',
                           'leftEyeLower1_252_x', 'leftEyeLower1_252_y', 'leftEyeLower1_256_x', 'leftEyeLower1_256_y',
                           'leftEyeLower1_341_x', 'leftEyeLower1_341_y', 'leftEyeLower1_463_x', 'leftEyeLower1_463_y',
                           'leftEyeUpper2_342_x', 'leftEyeUpper2_342_y', 'leftEyeUpper2_445_x', 'leftEyeUpper2_445_y',
                           'leftEyeUpper2_444_x', 'leftEyeUpper2_444_y', 'leftEyeUpper2_443_x', 'leftEyeUpper2_443_y',
                           'leftEyeUpper2_442_x', 'leftEyeUpper2_442_y', 'leftEyeUpper2_441_x', 'leftEyeUpper2_441_y',
                           'leftEyeUpper2_413_x', 'leftEyeUpper2_413_y', 'leftEyeLower2_446_x', 'leftEyeLower2_446_y',
                           'leftEyeLower2_261_x', 'leftEyeLower2_261_y', 'leftEyeLower2_448_x', 'leftEyeLower2_448_y',
                           'leftEyeLower2_449_x', 'leftEyeLower2_449_y', 'leftEyeLower2_450_x', 'leftEyeLower2_450_y',
                           'leftEyeLower2_451_x', 'leftEyeLower2_451_y', 'leftEyeLower2_452_x', 'leftEyeLower2_452_y',
                           'leftEyeLower2_453_x', 'leftEyeLower2_453_y', 'leftEyeLower2_464_x', 'leftEyeLower2_464_y',
                           'leftEyeLower3_372_x', 'leftEyeLower3_372_y', 'leftEyeLower3_340_x', 'leftEyeLower3_340_y',
                           'leftEyeLower3_346_x', 'leftEyeLower3_346_y', 'leftEyeLower3_347_x', 'leftEyeLower3_347_y',
                           'leftEyeLower3_348_x', 'leftEyeLower3_348_y', 'leftEyeLower3_349_x', 'leftEyeLower3_349_y',
                           'leftEyeLower3_350_x', 'leftEyeLower3_350_y', 'leftEyeLower3_357_x', 'leftEyeLower3_357_y',
                           'leftEyeLower3_465_x', 'leftEyeLower3_465_y', 'leftEyebrowUpper_383_x',
                           'leftEyebrowUpper_383_y', 'leftEyebrowUpper_300_x', 'leftEyebrowUpper_300_y',
                           'leftEyebrowUpper_293_x', 'leftEyebrowUpper_293_y', 'leftEyebrowUpper_334_x',
                           'leftEyebrowUpper_334_y', 'leftEyebrowUpper_296_x', 'leftEyebrowUpper_296_y',
                           'leftEyebrowUpper_336_x', 'leftEyebrowUpper_336_y', 'leftEyebrowUpper_285_x',
                           'leftEyebrowUpper_285_y', 'leftEyebrowUpper_417_x', 'leftEyebrowUpper_417_y',
                           'leftEyebrowLower_265_x', 'leftEyebrowLower_265_y', 'leftEyebrowLower_353_x',
                           'leftEyebrowLower_353_y', 'leftEyebrowLower_276_x', 'leftEyebrowLower_276_y',
                           'leftEyebrowLower_283_x', 'leftEyebrowLower_283_y', 'leftEyebrowLower_282_x',
                           'leftEyebrowLower_282_y', 'leftEyebrowLower_295_x', 'leftEyebrowLower_295_y',
                           'midwayBetweenEyes_168_x', 'midwayBetweenEyes_168_y', 'noseTip_1_x', 'noseTip_1_y',
                           'noseBottom_2_x', 'noseBottom_2_y', 'noseRightCorner_98_x', 'noseRightCorner_98_y',
                           'noseLeftCorner_327_x', 'noseLeftCorner_327_y', 'rightCheek_205_x', 'rightCheek_205_y',
                           'rightCheek_137_x', 'rightCheek_137_y', 'rightCheek_123_x', 'rightCheek_123_y',
                           'rightCheek_50_x', 'rightCheek_50_y', 'rightCheek_203_x', 'rightCheek_203_y',
                           'rightCheek_177_x', 'rightCheek_177_y', 'rightCheek_147_x', 'rightCheek_147_y',
                           'rightCheek_187_x', 'rightCheek_187_y', 'rightCheek_207_x', 'rightCheek_207_y',
                           'rightCheek_216_x', 'rightCheek_216_y', 'rightCheek_215_x', 'rightCheek_215_y',
                           'rightCheek_213_x', 'rightCheek_213_y', 'rightCheek_192_x', 'rightCheek_192_y',
                           'rightCheek_214_x', 'rightCheek_214_y', 'rightCheek_212_x', 'rightCheek_212_y',
                           'rightCheek_138_x', 'rightCheek_138_y', 'rightCheek_135_x', 'rightCheek_135_y',
                           'rightCheek_210_x', 'rightCheek_210_y', 'rightCheek_169_x', 'rightCheek_169_y',
                           'leftCheek_425_x', 'leftCheek_425_y', 'leftCheek_352_x', 'leftCheek_352_y',
                           'leftCheek_280_x', 'leftCheek_280_y', 'leftCheek_330_x', 'leftCheek_330_y',
                           'leftCheek_266_x', 'leftCheek_266_y', 'leftCheek_423_x', 'leftCheek_423_y',
                           'leftCheek_426_x', 'leftCheek_426_y', 'leftCheek_427_x', 'leftCheek_427_y',
                           'leftCheek_411_x', 'leftCheek_411_y', 'leftCheek_376_x', 'leftCheek_376_y',
                           'leftCheek_436_x', 'leftCheek_436_y', 'leftCheek_416_x', 'leftCheek_416_y',
                           'leftCheek_432_x', 'leftCheek_432_y', 'leftCheek_434_x', 'leftCheek_434_y',
                           'leftCheek_422_x', 'leftCheek_422_y', 'leftCheek_430_x', 'leftCheek_430_y',
                           'leftCheek_364_x', 'leftCheek_364_y', 'leftCheek_394_x', 'leftCheek_394_y',
                           'leftCheek_371_x', 'leftCheek_371_y'])

# Write the DataFrame to a CSV file
# csv_output_path = "Landmarks_Non_Stroke_Faces_withouts.csv"
csv_output_path = "Final_Non_Stroke.csv"
df.to_csv(csv_output_path, index=False)
