import tensorflow as tf
import numpy as np
from tqdm import tqdm
import datetime
import imageio
import os
import math
import csv
import copy
import sys
from sn import NO_OPS


class GAN(object):
    def __init__(self, sess, checkpoint_dir, sample_dir, time_path,
                 iteration, batch_size, data,
                 vis_freq, vis_num_h, vis_num_w,
                 latent,
                 generator, discriminator, n_dis,
                 metric_callbacks=None, metric_freq=None, metric_path=None,
                 time_freq=1000,
                 summary_freq=1,
                 g_lr=0.001, g_beta1=0.5, g_beta2=0.999,
                 d_lr=0.001, d_beta1=0.5, d_beta2=0.999,
                 save_checkpoint_freq=1000,
                 extra_checkpoint_freq=sys.maxint):
        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.time_path = time_path
        self.iteration = iteration
        self.batch_size = batch_size
        self.data = data
        self.vis_freq = vis_freq
        self.vis_num_h = vis_num_h
        self.vis_num_w = vis_num_w
        self.latent = latent
        self.generator = generator
        self.discriminator = discriminator
        self.n_dis = n_dis
        self.metric_callbacks = metric_callbacks
        self.metric_freq = metric_freq
        self.metric_path = metric_path
        self.time_freq = time_freq
        self.g_lr = g_lr
        self.g_beta1 = g_beta1
        self.g_beta2 = g_beta2
        self.d_lr = d_lr
        self.d_beta1 = d_beta1
        self.d_beta2 = d_beta2

        self.save_checkpoint_freq = save_checkpoint_freq
        self.extra_checkpoint_freq = extra_checkpoint_freq

        self.num_images = data.shape[0]
        self.image_dims = list(data.shape[1:])

        self.summary_freq = summary_freq

        self.EPS = 1e-8
        self.SN_OP = "spectral_norm_update_ops"
        self.MODEL_NAME = "model"

        if self.metric_callbacks is not None:
            for metric_callback in self.metric_callbacks:
                metric_callback.set_model(self)

        self.vis_random_latents = \
            self.latent.sample(vis_num_h * vis_num_w)

        self.log_latents = self.latent.sample(self.batch_size)

    def build(self):
        self.build_connection()
        self.build_loss()
        self.build_summary()
        self.build_metric()
        self.saver = tf.train.Saver()

    def build_metric(self):
        if self.metric_callbacks is not None:
            for metric_callback in self.metric_callbacks:
                metric_callback.build()

    def build_connection(self):
        self.z_pl = tf.placeholder(
            tf.float32, [None, self.latent.dim], name="z")
        self.real_image_pl = tf.placeholder(
            tf.float32, [None] + self.image_dims,
            name="real_image")

        # fake images
        self.fake_image_train_tf, _ = \
            self.generator.build(
                self.z_pl, train=True)
        self.fake_image_test_tf, _ = \
            self.generator.build(
                self.z_pl, train=False)

        self.d_real_image_train_tf, _, self.d_sigmas = \
            self.discriminator.build(
                self.real_image_pl,
                train=True,
                sn_op=self.SN_OP)
        self.d_fake_image_train_tf = \
            self.discriminator.build(
                self.fake_image_train_tf,
                train=True,
                sn_op=NO_OPS)[0]

        self.generator.print_layers()
        self.discriminator.print_layers()

    def build_loss(self):
        # generator
        self.g_loss_fake = -tf.reduce_mean(self.d_fake_image_train_tf)
        self.g_loss = self.g_loss_fake

        # discriminator
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.relu(1. + self.d_fake_image_train_tf))
        self.d_loss_real = tf.reduce_mean(
            tf.nn.relu(1. - self.d_real_image_train_tf))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        gradients = tf.gradients(
            self.discriminator.build(
                self.real_image_pl,
                train=True,
                sn_op=NO_OPS)[0],
            [self.real_image_pl])
        real_slopes = tf.reduce_sum(
            tf.square(gradients[0]),
            reduction_indices=[1, 2, 3])
        self.d_real_slopes = tf.sqrt(real_slopes + self.EPS)

        gradients = tf.gradients(
            self.discriminator.build(
                self.fake_image_train_tf,
                train=True,
                sn_op=NO_OPS)[0],
            [self.fake_image_train_tf])
        fake_slopes = tf.reduce_sum(
            tf.square(gradients[0]),
            reduction_indices=[1, 2, 3])
        self.d_fake_slopes = tf.sqrt(fake_slopes + self.EPS)

        self.g_op = \
            tf.train.AdamOptimizer(
                self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2)\
            .minimize(
                self.g_loss,
                var_list=self.generator.trainable_vars)

        self.d_op = \
            tf.train.AdamOptimizer(
                self.d_lr, beta1=self.d_beta1, beta2=self.d_beta2)\
            .minimize(
                self.d_loss,
                var_list=self.discriminator.trainable_vars)

        self.spectral_norm_update_ops = tf.get_collection(self.SN_OP)

    def build_summary(self):
        self.g_summary = []
        self.g_summary.append(tf.summary.scalar(
            "loss/g/fake", self.g_loss_fake))
        self.g_summary.append(tf.summary.scalar(
            "loss/g", self.g_loss))
        self.g_summary = tf.summary.merge(self.g_summary)

        self.d_summary = []
        self.d_summary.append(tf.summary.scalar(
            "loss/d/real", self.d_loss_real))
        self.d_summary.append(tf.summary.scalar(
            "loss/d/fake", self.d_loss_fake))
        self.d_summary.append(tf.summary.scalar(
            "loss/d", self.d_loss))
        self.d_summary.append(tf.summary.scalar(
            "d/real",
            tf.reduce_mean(self.d_real_image_train_tf)))
        self.d_summary.append(tf.summary.scalar(
            "d/fake",
            tf.reduce_mean(self.d_fake_image_train_tf)))

        for i in range(len(self.d_sigmas)):
            if isinstance(self.d_sigmas[i], list):
                for j in range(len(self.d_sigmas[i])):
                    self.d_summary.append(tf.summary.scalar(
                        "d/sigma{}_{}".format(i, j),
                        self.d_sigmas[i][j]))
            else:
                self.d_summary.append(tf.summary.scalar(
                    "d/sigma{}".format(i),
                    self.d_sigmas[i]))

        self.d_summary.append(tf.summary.scalar(
            "d/mean_real_slopes", tf.reduce_mean(self.d_real_slopes)))
        self.d_summary.append(tf.summary.scalar(
            "d/min_real_slopes", tf.math.reduce_min(self.d_real_slopes)))
        self.d_summary.append(tf.summary.scalar(
            "d/max_real_slopes", tf.math.reduce_max(self.d_real_slopes)))
        self.d_summary.append(tf.summary.scalar(
            "d/mean_fake_slopes", tf.reduce_mean(self.d_fake_slopes)))
        self.d_summary.append(tf.summary.scalar(
            "d/min_fake_slopes", tf.math.reduce_min(self.d_fake_slopes)))
        self.d_summary.append(tf.summary.scalar(
            "d/max_fake_slopes", tf.math.reduce_max(self.d_fake_slopes)))

        self.d_summary = tf.summary.merge(self.d_summary)

    def save(self, global_id, saver=None, checkpoint_dir=None):
        if saver is None:
            saver = self.saver
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        saver.save(
            self.sess,
            os.path.join(checkpoint_dir, self.MODEL_NAME),
            global_step=global_id)

    def load(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # In cases where people move the checkpoint directory to another place,
        # model path indicated by get_checkpoint_state will be wrong. So we
        # get the model name and then recontruct path using checkpoint_dir
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        global_id = int(ckpt_name[len(self.MODEL_NAME) + 1:])
        return global_id

    def _image_list_to_grid(self, image_list, num_row, num_col):
        assert num_row * num_col == image_list.shape[0]

        height, width, depth = self.image_dims
        image = np.zeros((num_row * height,
                          num_col * width,
                          depth))
        s_id = 0
        for row in range(num_row):
            for col in range(num_col):
                image[row * height: (row + 1) * height,
                      col * width: (col + 1) * width, :] = image_list[s_id]
                s_id += 1

        v_min = image.min() - self.EPS
        v_max = image.max() + self.EPS
        image = (image - v_min) / (v_max - v_min) * 255.0
        image = image.astype(np.uint8)

        print(v_min, v_max)

        return image

    def sample_from(self, z, noise_stddev=0.0):
        # print(z.shape)
        samples = []
        for i in range(int(math.ceil(float(z.shape[0]) / self.batch_size))):
            sub_samples = self.sess.run(
                self.fake_image_test_tf,
                feed_dict={self.z_pl: z[i * self.batch_size:
                                        (i + 1) * self.batch_size]})
            samples.append(sub_samples)
        return np.vstack(samples)

    def visualize(self, global_id):
        samples = self.sample_from(self.vis_random_latents)
        image = self._image_list_to_grid(
            samples, self.vis_num_h, self.vis_num_w)
        file_path = os.path.join(
            self.sample_dir,
            "global_id-{}.png".format(global_id))
        imageio.imwrite(file_path, image)

    def log_metric(self, global_id):
        if self.metric_callbacks is not None:
            metric = {}
            for metric_callback in self.metric_callbacks:
                metric.update(metric_callback.evaluate(-1, -1, global_id))
            if not os.path.isfile(self.metric_path):
                self.METRIC_FIELD_NAMES = ["global_id"]
                for k in metric:
                    self.METRIC_FIELD_NAMES.append(k)
                with open(self.metric_path, "wb") as csv_file:
                    writer = csv.DictWriter(
                        csv_file, fieldnames=self.METRIC_FIELD_NAMES)
                    writer.writeheader()
            elif not hasattr(self, "METRIC_FIELD_NAMES"):
                with open(self.metric_path, "rb") as csv_file:
                    reader = csv.DictReader(csv_file)
                    print("Load METRIC_FIELD_NAMES from the "
                          "existing metric file")
                    self.METRIC_FIELD_NAMES = reader.fieldnames

            with open(self.metric_path, "ab") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=self.METRIC_FIELD_NAMES)
                data = {
                    "global_id": global_id}
                metric_string = copy.deepcopy(metric)
                for k in metric_string:
                    if isinstance(metric[k], (float, np.float32, np.float64)):
                        metric_string[k] = "{0:.12f}".format(metric_string[k])
                data.update(metric_string)
                writer.writerow(data)
            for k in metric:
                if isinstance(metric[k], (int, long, float, complex,
                                          np.float32, np.float64)):
                    summary = tf.Summary(
                        value=[tf.Summary.Value(
                            tag="metric/" + k, simple_value=metric[k])])
                    self.summary_writer.add_summary(summary, global_id)

    def get_feed_dict(self):
        feed_dict = {}

        batch_data_id = np.random.choice(self.data.shape[0], self.batch_size)
        batch_image = self.data[batch_data_id]
        batch_z = self.latent.sample(self.batch_size)

        feed_dict[self.z_pl] = batch_z
        feed_dict[self.real_image_pl] = batch_image

        return feed_dict

    def train(self, restore=False):
        tf.global_variables_initializer().run()
        if restore is True:
            restore_global_id = self.load()
            print("Loaded from global_id {}".format(restore_global_id))
        else:
            restore_global_id = -1

        self.summary_writer = tf.summary.FileWriter(
            self.checkpoint_dir, self.sess.graph)

        if self.metric_callbacks is not None:
            for metric_callback in self.metric_callbacks:
                metric_callback.load()

        for global_id in tqdm(range(restore_global_id + 1, self.iteration)):
            if ((global_id + 1) % self.time_freq == 0 or
                    global_id == self.iteration - 1 or
                    global_id == 0):
                with open(self.time_path, "a") as f:
                    time = datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S.%f")
                    f.write("iter {} starts: {}\n".format(global_id, time))

            for i in range(self.n_dis - 1):
                feed_dict = self.get_feed_dict()
                self.sess.run(
                    [self.d_op],
                    feed_dict=feed_dict)
                self.sess.run(self.spectral_norm_update_ops)

            feed_dict = self.get_feed_dict()
            summary_result, _ = self.sess.run(
                [self.d_summary, self.d_op],
                feed_dict=feed_dict)
            if (global_id + 1) % self.summary_freq == 0:
                self.summary_writer.add_summary(summary_result, global_id)
            self.sess.run(self.spectral_norm_update_ops)

            summary_result, _ = self.sess.run(
                [self.g_summary, self.g_op],
                feed_dict=feed_dict)
            if (global_id + 1) % self.summary_freq == 0:
                self.summary_writer.add_summary(summary_result, global_id)

            if (global_id + 1) % self.vis_freq == 0:
                self.visualize(global_id)

            if (global_id + 1) % self.metric_freq == 0:
                self.log_metric(global_id)

            if (global_id + 1) % self.extra_checkpoint_freq == 0:
                self.visualize(global_id)
                self.log_metric(global_id)
                saver = tf.train.Saver()
                checkpoint_dir = os.path.join(
                    self.checkpoint_dir,
                    "global_id-{}".format(global_id))
                self.save(global_id, saver, checkpoint_dir)

            if ((global_id + 1) % self.save_checkpoint_freq == 0 or
                    global_id == self.iteration - 1):
                self.save(global_id)

            if ((global_id + 1) % self.time_freq == 0 or
                    global_id == self.iteration - 1 or
                    global_id == 0):
                with open(self.time_path, "a") as f:
                    time = datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S.%f")
                    f.write("iter {} ends: {}\n".format(global_id, time))


if __name__ == "__main__":
    from latent import GaussianLatent
    from network import Generator, Discriminator
    from load_data import load_celeba
    from metric import InceptionScore, FrechetInceptionDistance

    data = load_celeba("../data/CelebA")
    _, height, width, depth = data.shape
    print(data.min(), data.max())

    latent = GaussianLatent(dim=128, loc=0.0, scale=1.0)

    generator = Generator(
        output_width=width, output_height=height, output_depth=depth, mg=8)
    discriminator = Discriminator(sn_mode="SN")

    checkpoint_dir = "./test/checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    sample_dir = "./test/sample"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    time_path = "./test/time.txt"
    iteration = 400000
    batch_size = 64
    vis_freq = 200
    vis_num_h = 10
    vis_num_w = 10

    g_lr = 0.0002
    g_beta1 = 0.5
    g_beta2 = 0.999
    d_lr = 0.0002
    d_beta1 = 0.5
    d_beta2 = 0.999

    n_dis = 1
    metric_freq = 10000
    metric_path = "./test/metrics.csv"

    run_config = tf.ConfigProto()
    with tf.Session(config=run_config) as sess:
        metric_callbacks = [
            InceptionScore(sess=sess),
            FrechetInceptionDistance(sess=sess, real_images=data)]
        gan = GAN(
            sess=sess,
            checkpoint_dir=checkpoint_dir,
            sample_dir=sample_dir,
            time_path=time_path,
            iteration=iteration,
            batch_size=batch_size,
            data=data,
            vis_freq=vis_freq,
            vis_num_h=vis_num_h,
            vis_num_w=vis_num_w,
            latent=latent,
            generator=generator,
            discriminator=discriminator,
            n_dis=n_dis,
            metric_callbacks=metric_callbacks,
            metric_freq=metric_freq,
            metric_path=metric_path,
            g_lr=g_lr,
            g_beta1=g_beta1,
            g_beta2=g_beta2,
            d_lr=d_lr,
            d_beta1=g_beta1,
            d_beta2=g_beta2)
        gan.build()
        gan.train()
