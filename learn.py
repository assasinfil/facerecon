import os

import cv2
import numpy as np
import mediapipe as mp
from pyransac3d import rodrigues_rot


def get_face_coordinates(face_landmarks, image):
    x_max = 0
    y_max = 0
    h, w, c = image.shape
    x_min = w
    y_min = h
    for landmark in face_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
    return x_min, y_min, x_max, y_max


def face_landmarks_to_array(face_landmarks):
    face = np.array(
        [[res.x, res.y, res.z] for res in face_landmarks.landmark]) if face_landmarks.landmark else np.zeros(
        468 * 3).reshape(468, 3)
    center_point = [0, 0, 0]
    for point in face:
        center_point += point / len(face)
    face = rodrigues_rot(face, center_point, [0, 0, 1])
    # face[:, 2] -= 1
    return face


def save_face_vectors(face, face_id):
    np.save(f"faces/{face_id}", face)


def save_face_image(face_landmarks, face_id, image):
    x_min, y_min, x_max, y_max = get_face_coordinates(face_landmarks, image)
    try:
        cv2.imwrite(f"images/{face_id}.jpg", image[y_min:y_max, x_min:x_max])
    except:
        print("error saving face")


if __name__ == '__main__':
    IMAGE_FILES = os.listdir("images")
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5) as face_mesh:
        names = list()
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(f"images/{file}")
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                face = face_landmarks_to_array(face_landmarks)
                save_face_vectors(face, idx)
                names.append(f"{idx} {file.split('.')[0]}\n")
        with open("names.txt", 'w') as f:
            f.writelines(names)
