import cv2
import os


# search file from base dir to the end
def search_file(base_dir, video_name):
    for root, dirs, files in os.walk(base_dir):
        if video_name in files:
            return os.path.join(root, video_name)


def video_to_img(base_dir, video_name, frame_interval=100):
    # create necessery dictionary data/imgs
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    imgs_dir = os.path.join(data_dir, "imgs")
    os.makedirs(imgs_dir, exist_ok=True)

    vname = os.path.splitext(video_name)[0]

    # create a dictionary with the same name of video name
    output_dir = os.path.join(imgs_dir, vname)
    os.makedirs(output_dir, exist_ok=True)

    # get path of video
    video_path = search_file(base_dir, video_name)

    # open vedio
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: could not open vedio file {video_path}")
        return

    frame_count = 0
    image_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            image_path = os.path.join(output_dir, f"{vname}_{image_count}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved: {image_path}")
            image_count += 1

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total images saved: {image_count}")


def main():
    base_dir = os.getcwd()
    video_name = "video"
    if not video_name.endswith(".mp4"):
        video_name += ".mp4"
    video_to_img(base_dir, video_name)


if __name__ == "__main__":
    main()
