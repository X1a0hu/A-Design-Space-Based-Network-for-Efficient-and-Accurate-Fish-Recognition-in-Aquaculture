import json
import os
import numpy as np


class Label:

    def __init__(self, num_points, shape, label=0, box=np.zeros(4, dtype=np.float32)):
        self.shape = shape
        self.label = label
        self.box = box
        self.points = np.zeros((num_points, 3), dtype=np.float32)

    def update_points(self, points, index):
        self.points[index] = points

    def nomorlize(self):
        self.box[::2] /= self.shape[0]
        self.box[1::2] /= self.shape[1]

        self.points[:, 0] /= self.shape[0]
        self.points[:, 1] /= self.shape[1]

    def write(self):
        self.nomorlize()
        return [self.label, *self.box, *self.points.reshape(-1)]


class Converter:
    def __init__(self, classes, points):
        self.classes = np.array(classes)
        self.points = np.array(points, dtype=object)

    def keypoint_convert(self, data):
        shapes = data.get("shapes", [])
        width = data.get("imageWidth")
        height = data.get("imageHeight")
        pred_dict = {}

        for shape in shapes:
            label = shape.get("label")
            points = shape.get("points", [])
            group_id = shape.get("group_id")
            description = shape.get("description")

            if group_id not in pred_dict:
                pred_dict[group_id] = []

            pred_dict[group_id].append(
                {
                    "label": label,
                    "points": points,
                    "description": description,
                }
            )

        content = []
        for dict_list in pred_dict.values():
            label_list = np.array([d["label"] for d in dict_list])
            cls_indice = self.get_class_indice(label_list)

            if cls_indice.size == 0:
                continue

            box = self.get_box(dict_list[cls_indice[0]])
            label = Label(len(self.points), [width, height], cls_indice[1], box)

            point_indice = self.get_point_indices(label_list)
            for point, index in point_indice:
                points = dict_list[point]["points"][0]
                points.append(float(dict_list[point]["description"]))

                label.update_points(points, index)

            content.append(label.write())

        return content

    def write(self, content, output_file):
        with open(output_file, "w") as f:
            for row in content:
                line = " ".join(map(str, row)) + "\n"
                f.write(line)

    def get_box(self, dict: dict):
        (x1, y1), (x2, y2) = dict["points"]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y1 - y2
        return np.array([x_center, y_center, width, height])

    def get_class_indice(self, label_list):
        matches = np.isin(label_list, self.classes)
        return np.array(
            [
                (i, np.where(self.classes == label)[0][0])
                for i, label in enumerate(label_list)
                if matches[i]
            ]
        )[0]

    def get_point_indices(self, label_list):
        matches = np.isin(label_list, self.points)
        return np.array(
            [
                (i, np.where(self.points == label)[0][0])
                for i, label in enumerate(label_list)
                if matches[i]
            ]
        )


if __name__ == "__main__":
    base_dir = os.getcwd()
    json_file = os.path.join(base_dir, "test.json")

    classes = ["fish"]
    points = ["left_eye", "right_eye", "back", "body", "tail"]

    con = Converter(classes, points)

    with open(json_file, "r") as f:
        data = json.load(f)
        con.keypoint_convert(data)
