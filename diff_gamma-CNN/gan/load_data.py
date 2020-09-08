import tarfile
import os
import cPickle
import numpy as np


def load_celeba(path):
    data = np.load(os.path.join(path, "celeba.npy"))
    data = data.astype(np.float64)
    data = data / 255. * 2.0 - 1.0  # -1~1
    return data


def load_stl10(path):
    data = np.load(os.path.join(path, "stl10.npy"))
    data = data.astype(np.float64)
    data = data / 255. * 2.0 - 1.0  # -1~1
    return data


def unpickle_cifar10(file):
    return cPickle.load(file)


def load_cifar10(path):
    tar = tarfile.open(os.path.join(path, "cifar-10-python.tar.gz"))
    data = []
    for i in range(1, 6):
        file = tar.extractfile(
            os.path.join("cifar-10-batches-py", "data_batch_{}".format(i)))
        sub_data = unpickle_cifar10(file)["data"]
        assert list(sub_data.shape) == [10000, 3072]
        assert sub_data.dtype == np.uint8
        data.append(sub_data)

    data = np.concatenate(data, axis=0)
    assert list(data.shape) == [50000, 3072]

    data = np.reshape(data, [50000, 3, 32, 32])
    data = np.transpose(data, [0, 2, 3, 1])
    data = data.astype(np.float64)
    data = data / 255. * 2.0 - 1.0  # -1~1
    return data
