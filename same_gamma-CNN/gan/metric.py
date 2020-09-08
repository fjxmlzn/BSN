import numpy as np
import os
import functools
import tensorflow as tf
from tensorflow.python.ops import array_ops
tfgan = tf.contrib.gan


class Metric(object):
    def __init__(self, sess):
        self.sess = sess
        self.model = None

    def set_model(self, model):
        self.model = model

    def build(self):
        pass

    def load(self):
        pass

    def evaluate(self, epoch_id, batch_id, global_id):
        raise NotImplementedError


class FrechetInceptionDistance(Metric):
    """
    Adapted from:
    https://github.com/tsc2017/Frechet-Inception-Distance/blob/master/fid.py
    """

    def __init__(self, real_images, batch_size=64,
                 num_gen_images=10000, num_real_images=5000,
                 image_min=-1, image_max=1,
                 *args, **kwargs):
        super(FrechetInceptionDistance, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_gen_images = num_gen_images
        self.num_real_images = num_real_images
        self.image_min = image_min
        self.image_max = image_max

        self.real_images = real_images
        if not (np.min(self.real_images) >= self.image_min and
                np.max(self.real_images) <= self.image_max):
            raise Exception("range of pixels incorrect")

        id_ = np.random.permutation(real_images.shape[0])
        self.real_images = self.real_images[id_[:num_real_images]]
        self.real_images = self.transform_image(self.real_images)

    def build(self):
        self.inception_images_pl = \
            tf.placeholder(
                tf.float32,
                [None, None, None, 3],
                name="fid_images")

        self.activations1_pl = \
            tf.placeholder(
                tf.float32,
                [None, None],
                name="fid_activations1")
        self.activations2_pl = \
            tf.placeholder(
                tf.float32,
                [None, None],
                name="fid_activations2")
        self.fcd = tfgan.eval.frechet_classifier_distance_from_activations(
            self.activations1_pl,
            self.activations2_pl)

    def load(self):
        size = 299
        images = tf.compat.v1.image.resize_bilinear(
            self.inception_images_pl,
            [size, size])
        generated_images_list = array_ops.split(
            images, num_or_size_splits=1)
        activations = tf.map_fn(
            fn=functools.partial(tfgan.eval.run_inception,
                                 output_tensor="pool_3:0"),
            elems=array_ops.stack(generated_images_list),
            parallel_iterations=8,
            back_prop=False,
            swap_memory=True,
            name="fid_run_classifier")
        self.activations = array_ops.concat(array_ops.unstack(activations), 0)

    def get_inception_activations(self, inps):
        n_batches = int(np.ceil(float(inps.shape[0]) / self.batch_size))
        act = []
        for i in range(n_batches):
            inp = inps[i * self.batch_size:
                       (i + 1) * self.batch_size] / 255. * 2 - 1
            sub_act = self.sess.run(
                self.activations,
                feed_dict={self.inception_images_pl: inp})
            act.append(sub_act)
        act = np.concatenate(act, axis=0)
        return act

    def activation2distance(self, act1, act2):
        return self.sess.run(
            self.fcd,
            feed_dict={self.activations1_pl: act1,
                       self.activations2_pl: act2})

    def transform_image(self, images):
        images = (images - self.image_min) / (self.image_max - self.image_min)
        images = images * 255.

        return images

    def evaluate(self, epoch_id, batch_id, global_id):
        latents = self.model.latent.sample(self.num_gen_images)
        images = self.model.sample_from(latents)

        if not (np.min(images) >= self.image_min and
                np.max(images) <= self.image_max):
            raise Exception("range of pixels incorrect")
        images = self.transform_image(images)

        act1 = self.get_inception_activations(self.real_images)
        act2 = self.get_inception_activations(images)
        fid = self.activation2distance(act1, act2)

        return {"fid": fid}


class InceptionScore(Metric):
    """
    Adapted from:
    https://github.com/tsc2017/Inception-Score/blob/master/inception_score.py
    """

    def __init__(self, batch_size=64, num_splits=10, num_images=50000,
                 image_min=-1, image_max=1,
                 inception_url=("http://download.tensorflow.org/models/"
                                "frozen_inception_v1_2015_12_05.tar.gz"),
                 inception_frozen_graph="inceptionv1_for_inception_score.pb",
                 *args, **kwargs):
        super(InceptionScore, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_splits = num_splits
        self.num_images = num_images
        self.image_min = image_min
        self.image_max = image_max
        self.inception_url = inception_url
        self.inception_frozen_graph = inception_frozen_graph

    def build(self):
        self.inception_images_pl = \
            tf.placeholder(
                tf.float32,
                [None, None, None, 3],
                name="inception_score_images")

    def load(self):
        size = 299
        images = tf.compat.v1.image.resize_bilinear(
            self.inception_images_pl,
            [size, size])
        generated_images_list = array_ops.split(
            images, num_or_size_splits=1)
        logits = tf.map_fn(
            fn=functools.partial(
                tfgan.eval.run_inception,
                default_graph_def_fn=functools.partial(
                    tfgan.eval.get_graph_def_from_url_tarball,
                    self.inception_url,
                    self.inception_frozen_graph,
                    os.path.basename(self.inception_url)),
                output_tensor="logits:0"),
            elems=array_ops.stack(generated_images_list),
            parallel_iterations=8,
            back_prop=False,
            swap_memory=True,
            name='inception_score_run_classifier')
        self.logits = array_ops.concat(array_ops.unstack(logits), 0)

    def get_inception_probs(self, inps):
        n_batches = int(np.ceil(float(inps.shape[0]) / self.batch_size))
        preds = []
        for i in range(n_batches):
            inp = inps[i * self.batch_size:
                       (i + 1) * self.batch_size] / 255. * 2 - 1
            sub_preds = self.sess.run(
                self.logits,
                {self.inception_images_pl: inp})
            sub_preds = sub_preds[:, :1000]
            preds.append(sub_preds)
        preds = np.concatenate(preds, axis=0)
        preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
        return preds

    def preds2score(self, preds, splits=10):
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):
                         ((i + 1) * preds.shape[0] // splits)]
            kl = part * (np.log(part) -
                         np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)

    def transform_image(self, images):
        images = (images - self.image_min) / (self.image_max - self.image_min)
        images = images * 255.

        return images

    def evaluate(self, epoch_id, batch_id, global_id):
        latents = self.model.latent.sample(self.num_images)
        images = self.model.sample_from(latents)

        if not (np.min(images) >= self.image_min and
                np.max(images) <= self.image_max):
            raise Exception("range of pixels incorrect")

        images = self.transform_image(images)

        preds = self.get_inception_probs(images)
        mean, std = self.preds2score(preds, self.num_splits)

        return {"inception_score_mean": mean,
                "inception_score_std": std}
