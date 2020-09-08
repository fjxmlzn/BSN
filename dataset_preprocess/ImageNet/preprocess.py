import cv2
import tarfile
import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp


def process_image(image_str, save_path, target_size):
    image_np = cv2.imdecode(
        np.fromstring(image_str, np.uint8),
        cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    height, width, depth = image_np.shape

    # crop
    size = min(height, width)
    left_margin = (width - size) // 2
    right_margin = width - size - left_margin
    top_margin = (height - size) // 2
    bottom_margin = height - size - top_margin

    image_np = image_np[top_margin: height - bottom_margin,
                        left_margin: width - right_margin]

    # resize
    image_np = cv2.resize(image_np, (target_size, target_size))

    # save
    np.save(save_path, image_np)


def worker(queue):
    while True:
        item = queue.get()
        if item is None:
            break
        process_image(
            item["image_str"],
            item["save_path"],
            item["target_size"])


def preprocess_ilsvrc2012(path, save_path, num_processes=50, target_size=128):
    queue = mp.Queue(num_processes * 2)
    processes = []
    for i in range(num_processes):
        process = mp.Process(target=worker, args=(queue,))
        process.start()
        processes.append(process)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tar = tarfile.open(os.path.join(path, "ILSVRC2012_img_train.tar"))

    for sub_tar_i, sub_tar in enumerate(tqdm(tar.getmembers())):

        sub_save_path = os.path.join(
            save_path,
            os.path.splitext(sub_tar.name)[0])
        if not os.path.exists(sub_save_path):
            os.makedirs(sub_save_path)

        sub_tar = tar.extractfile(sub_tar)
        sub_tar = tarfile.open(fileobj=sub_tar)

        for image in tqdm(sub_tar.getmembers(), leave=False):

            image_save_path = os.path.join(
                sub_save_path,
                os.path.splitext(image.name)[0]) + ".npy"

            image = sub_tar.extractfile(image)
            image_str = image.read()

            queue.put({"image_str": image_str,
                       "save_path": image_save_path,
                       "target_size": target_size})

        sub_tar.close()
    tar.close()

    for i in range(num_processes):
        queue.put(None)

    for process in processes:
        process.join()


if __name__ == "__main__":
    preprocess_ilsvrc2012(".", "ILSVRC2012")
