import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp


class NpyFolderDataset(object):
    def __init__(self, root, buffer_size=2000, num_processes=50):
        self.root = root
        self.buffer_size = buffer_size
        self.num_processes = num_processes

        self.files = []
        self.labels = []
        self.num_per_class = []
        self.class_folders = os.listdir(root)
        self.class_folders = [os.path.join(root, folder)
                              for folder in self.class_folders]
        self.class_folder = [folder
                             for folder in self.class_folders
                             if os.path.isdir(folder)]

        for folder_i, folder in enumerate(tqdm(self.class_folders)):
            sub_files = os.listdir(folder)
            sub_files = [os.path.join(folder, file) for file in sub_files]
            sub_files = [file for file in sub_files if file.endswith(".npy")]
            self.files.extend(sub_files)
            self.labels.extend([folder_i] * len(sub_files))
            self.num_per_class.append(len(sub_files))

        self.num_classes = len(self.num_per_class)
        self.num_images = np.sum(self.num_per_class)

        self.image_buffer = mp.Queue(self.buffer_size)
        self.running_flag = mp.Value("i", 1)
        self.processes = []
        for i in range(num_processes):
            process = mp.Process(
                target=self.data_loader,
                args=(self.running_flag,
                      self.image_buffer,
                      self.files,
                      self.transform))
            process.start()
            self.processes.append(process)

        sample_data = self.sample_batch(1)
        self.image_height = sample_data.shape[1]
        self.image_width = sample_data.shape[2]
        self.image_depth = sample_data.shape[3]

    @staticmethod
    def data_loader(running_flag, image_buffer, files, transform):
        np.random.seed(os.getpid())
        while running_flag.value:
            file_id = np.random.choice(len(files))
            image = np.load(files[file_id])
            image = transform(image)
            image_buffer.put(image)
        image_buffer.cancel_join_thread()

    def stop_data_loader(self):
        self.running_flag.value = 0
        while not self.image_buffer.empty():
            try:
                self.image_buffer.get_nowait()
            except mp.Queue.Empty:
                pass

        for process in self.processes:
            #process.join()
            process.terminate()

    @staticmethod
    def transform(self, image):
        return image

    def sample_batch(self, batch_size):
        images = []
        for i in range(batch_size):
            images.append(self.image_buffer.get())
        images = np.stack(images, axis=0)
        return images


class ILSVRC2012Dataset(NpyFolderDataset):
    def __init__(self, *args, **kwargs):
        super(ILSVRC2012Dataset, self).__init__(*args, **kwargs)

        if self.num_classes != 1000:
            raise Exception("Number of classes should be 1000")

        if self.num_images != 1281167:
            raise Exception("Number of images should be 1281167")

    @staticmethod
    def transform(images):
        images = images.astype(np.float32)
        images = images / 255.0 * 2.0 - 1.0
        return images


if __name__ == "__main__":
    import imageio

    dataset = ILSVRC2012Dataset("ILSVRC2012")
    print(dataset.num_classes)
    print(dataset.num_images)
    print(dataset.num_per_class)

    images = dataset.sample_batch(10)
    print(np.min(images), np.max(images))

    dataset.stop_data_loader()

    for i in range(10):
        imageio.imwrite(
            "test{}.png".format(i),
            images[i])
