import os
import cv2
import numpy as np
import shutil


def color(index):
    import matplotlib.pyplot as plt

    colors = [
        (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in plt.cm.tab10.colors
    ]
    return colors[index % len(colors)]


# the base_name is out of extension like "input" not "input.txt"
def get_unique_filename(output_dir, base_name, extension):
    counter = 1
    output_file = os.path.join(output_dir, f"{base_name}{extension}")

    while os.path.exists(output_file):
        output_file = os.path.join(output_dir, f"{base_name}{counter}{extension}")
        counter += 1

    return output_file


def rename_in_order(image_dir, label_dir, output_dir, name="fish", begin=1):
    output_image_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    image_files = {os.path.splitext(file)[0]: file for file in os.listdir(image_dir)}
    label_files = {os.path.splitext(file)[0]: file for file in os.listdir(label_dir)}
    for base_name in image_files.keys():
        if base_name in label_files:
            new_base_name = f"{name}_{begin:05d}"
            begin += 1

            image_file = os.path.join(image_dir, image_files[base_name])
            label_file = os.path.join(label_dir, label_files[base_name])

            new_image_file = os.path.join(output_image_dir, f"{new_base_name}.jpg")
            new_label_file = os.path.join(output_label_dir, f"{new_base_name}.txt")

            shutil.copy(image_file, new_image_file)
            shutil.copy(label_file, new_label_file)
    print(f"Files saved to {output_dir}")


def mask_to_yolo(mask_dir, output_dir, label=0):
    max_points = 30

    for file_name in os.listdir(mask_dir):
        output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".txt")
        with open(output_file, "w") as file:
            mask = cv2.imread(os.path.join(mask_dir, file_name), cv2.IMREAD_GRAYSCALE)
            height, width = mask.shape
            _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            # Detect borders
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if len(contour) > max_points:
                    indices = np.linspace(0, len(contour) - 1, max_points, dtype=int)
                    contour = contour[indices]

                file.write(f"{label}")
                for point in contour / [width, height]:
                    file.write(f" {point[0][0]} {point[0][1]}")
                file.write("\n")

    print(f"Saved to {output_dir}")


def data_divider(image_dir, label_dir, output_dir, interval=10):
    train_image_dir = os.path.join(output_dir, "images", "train")
    train_label_dir = os.path.join(output_dir, "labels", "train")
    val_image_dir = os.path.join(output_dir, "images", "val")
    val_label_dir = os.path.join(output_dir, "labels", "val")
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    image_files = {
        os.path.splitext(file)[0]: file
        for file in os.listdir(image_dir)
        if file.endswith((".jpg", ".png"))
    }
    label_files = {
        os.path.splitext(file)[0]: file
        for file in os.listdir(label_dir)
        if file.endswith(".txt")
    }

    idx = 1
    for base_name, image_file in image_files.items():
        if base_name in label_files:
            src_image_file = os.path.join(image_dir, image_file)
            src_label_file = os.path.join(label_dir, label_files[base_name])

            if idx % interval == 0:
                dst_image_file = os.path.join(val_image_dir, image_file)
                dst_label_file = os.path.join(val_label_dir, label_files[base_name])
            else:
                dst_image_file = os.path.join(train_image_dir, image_file)
                dst_label_file = os.path.join(train_label_dir, label_files[base_name])

            idx += 1
            print(
                f"Copy {src_image_file} to {dst_image_file}. Copy {src_label_file} to {dst_label_file}"
            )
            shutil.copy(src_image_file, dst_image_file)
            shutil.copy(src_label_file, dst_label_file)

    print(f"Data saved to {output_dir}")


def image_converter(image_dir, output_dir):
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(image_dir):
        with Image.open(os.path.join(image_dir, file), "r") as img:
            img = img.convert("RGB")
            img.save(
                os.path.join(output_dir, os.path.splitext(file)[0] + ".jpg"),
                format="JPEG",
            )


def id_label(label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(label_dir):
        output_file = os.path.join(output_dir, file_name)

        with open(os.path.join(label_dir, file_name), "r") as f:
            label = f.readline().split(" ")[0]

        with open(output_file, "w") as of:
            of.write(label)


def id_exctract(input_dir, output_dir):
    output_image_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for dir_1 in os.listdir(input_dir):
        dir = os.path.join(input_dir, dir_1)
        for dir_2 in os.listdir(dir):
            sub_dir = os.path.join(dir, dir_2)
            for file_name in os.listdir(sub_dir):
                base_name, extend = os.path.splitext(file_name)
                if extend == ".jpg":
                    image_file = os.path.join(sub_dir, base_name + ".jpg")
                    label_file = os.path.join(sub_dir, base_name + ".txt")

                    dst_image_file = os.path.join(output_image_dir, base_name + ".jpg")
                    dst_label_file = os.path.join(output_label_dir, base_name + ".txt")

                    if os.path.exists(image_file) and os.path.exists(label_file):
                        shutil.copy(image_file, dst_image_file)
                        shutil.copy(label_file, dst_label_file)
