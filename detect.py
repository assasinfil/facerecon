import os

import cv2
import mediapipe as mp
import numpy as np
from pyransac3d import rodrigues_rot

MAX_DISTANCE = 0
THRESHOLD = 5


def get_face_coordinates(face_landmarks, image):
    h, w, _ = image.shape
    x_max = 0
    y_max = 0
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


def get_face_center(x_min, y_min, x_max, y_max):
    return int((x_max - x_min) / 2) + x_min, int((y_max - y_min) / 2) + y_min


class Face:
    def __init__(self, face_landmarks):
        self.face_landmarks = face_landmarks
        self.name = ""
        self.x = -1
        self.y = -1
        self.tracked = False
        self.face_points = None

    def show(self, image):
        # x_min, y_min, x_max, y_max = get_face_coordinates(self.face_landmarks, image)
        cv2.putText(image, self.name, (self.x - 60, self.y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    def distance(self, face_landmarks):
        face_points1 = self.face_points
        # if self.face_landmarks is not None:
        #     face_points1 = face_landmarks_to_array(self.face_landmarks)
        face_points2 = face_landmarks_to_array(face_landmarks)
        distances = [np.sqrt((face_points1[i][0] - face_points2[i][0]) ** 2 + (
                face_points1[i][1] - face_points2[i][1]) ** 2) for i in
                     range(len(face_points1))]
        return sum(distances)


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


def point_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def load_names():
    names = dict()
    with open("names.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            face_id, name = line.split(" ")
            names[face_id] = name[:-1]
    return names


def load_faces_data(names):
    faces = list()
    face_vectors_files = os.listdir("faces")
    for file in face_vectors_files:
        if file.endswith(".npy"):
            try:
                face_landmarks = np.load(f"faces/{file}")
                face = Face(None)
                face.face_points = face_landmarks
                name = names[file.split(".")[0]]
                if name is None:
                    face.name = file.split(".")[0]
                else:
                    face.name = name
                faces.append(face)
            except:
                print(f"error loading face {file}")
                continue
    return faces


if __name__ == '__main__':
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    tracked_faces = list()
    names = load_names()
    faces = load_faces_data(names)
    with mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=False, min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image.flags.writeable = False
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[..., 0] = clahe.apply(lab[..., 0])
            equ = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            results = face_mesh.process(equ)
            image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for tracked_face in tracked_faces:
                    tracked_face.tracked = False
                for face_landmarks in results.multi_face_landmarks:
                    found = False
                    x_min, y_min, x_max, y_max = get_face_coordinates(face_landmarks, image)
                    x, y = get_face_center(x_min, y_min, x_max, y_max)
                    # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.circle(image, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
                    distances = [point_distance(x, y, tracked_face.x,
                                                tracked_face.y) for tracked_face in tracked_faces]
                    if len(distances) > 0:
                        min_distance_index = np.argmin(distances)
                        if distances[min_distance_index] < THRESHOLD:
                            tracked_faces_indexes_to_remove = [i for i, v in enumerate(distances) if
                                                               distances[min_distance_index] < v < THRESHOLD]
                            tracked_face_founded = tracked_faces[min_distance_index]
                            tracked_face_founded.x = x
                            tracked_face_founded.y = y
                            tracked_face_founded.face_landmarks = face_landmarks
                            tracked_face_founded.show(image)
                            tracked_face_founded.tracked = True

                            for index in tracked_faces_indexes_to_remove:
                                tracked_faces.pop(index)
                            found = True

                    if not found:
                        distances = [face.distance(face_landmarks) for face in faces]
                        if len(distances) > 0:
                            min_distance_index = np.argmin(distances)
                            if distances[min_distance_index] < THRESHOLD:
                                face = faces[min_distance_index]
                                x_min, y_min, x_max, y_max = get_face_coordinates(face_landmarks, image)
                                face.x, face.y = get_face_center(x_min, y_min, x_max, y_max)
                                face.tracked = True
                                tracked_faces.append(face)
                                print(f"Added face to tracking, distance: {distances[min_distance_index]}")

                                break
                for tracked_face in tracked_faces:
                    if tracked_face.tracked is False:
                        tracked_faces.remove(tracked_face)

            cv2.imshow('MediaPipe Face Mesh', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
