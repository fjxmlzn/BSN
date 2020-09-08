import tarfile
import os
import numpy as np
import cv2


def preprocess_stl10(path, target_size=48):
    tar = tarfile.open(os.path.join(path, "stl10_binary.tar.gz"))
    file = tar.extractfile(os.path.join("stl10_binary", "unlabeled_X.bin"))
    images = np.frombuffer(file.read(), dtype=np.uint8)
    images = np.reshape(images, (-1, 3, 96, 96))
    images = np.transpose(images, (0, 3, 2, 1))
    assert list(images.shape) == [100000, 96, 96, 3]
    new_images = [cv2.resize(image, (target_size, target_size))
                  for image in images]
    new_images = np.stack(new_images, axis=0)
    np.save("stl10.npy", new_images)
    return new_images


if __name__ == "__main__":
    preprocess_stl10(".")
