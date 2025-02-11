import numpy as np
import cv2


class BoundingBox:
    def __init__(self, x, y, w, h, label):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label


class Node:
    def __init__(self, bounding_box):
        self.bounding_box = bounding_box
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def pre_order(self):
        return sum([child.pre_order() for child in self.children], [self.bounding_box])


def calculate_corners(x, y, w, h, r):
    x, y, w, h, r = float(x), float(y), float(w), float(h), float(r)
    r = np.deg2rad(r)

    corners = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
    )

    rotation_matrix = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])

    rotated_corners = np.dot(corners, rotation_matrix.T) + [x, y]
    return rotated_corners


def calculate_midpoint(p1, p2):
    return (p1 + p2) / 2


def calculate_midpoints(xyxyxyxy):
    a, b, c, d = xyxyxyxy

    midpoints = [
        calculate_midpoint(a, b),
        calculate_midpoint(b, c),
        calculate_midpoint(c, d),
        calculate_midpoint(d, a),
    ]
    return midpoints


def calculate_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


# def preprocess_photography_image(image_path, output_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     contrast_image = clahe.apply(image)
#     cv2.imwrite(output_path, contrast_image)


def preprocess_photography_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(output_path, image)
