import tensorflow as tf
from op import linear, batch_norm, deconv2d, conv2d, lrelu
import os


class Network(object):
    def __init__(self, scope_name):
        self.scope_name = scope_name

    def build(self, input):
        raise NotImplementedError

    @property
    def all_vars(self):
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.scope_name)

    @property
    def trainable_vars(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope_name)

    def get_variable(self, name):
        for var in self.all_vars:
            if var.name == name:
                return var
        return None

    def print_layers(self):
        print("Layers of {}".format(self.scope_name))
        print(self.all_vars)

    def save(self, sess, folder):
        saver = tf.train.Saver(self.all_vars)
        path = os.path.join(folder, "model.ckpt")
        saver.save(sess, path)

    def load(self, sess, folder):
        saver = tf.train.Saver(self.all_vars)
        path = os.path.join(folder, "model.ckpt")
        saver.restore(sess, path)


class Generator(Network):
    def __init__(self, output_width, output_height, output_depth, mg,
                 scope_name="generator", *args, **kwargs):
        super(Generator, self).__init__(scope_name=scope_name, *args, **kwargs)
        self.output_width = output_width
        self.output_height = output_height
        self.output_depth = output_depth
        self.mg = mg

    def build(self, z, train):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(z)[0]

            layers = [z]

            with tf.variable_scope("layer0"):
                layers.append(linear(layers[-1], self.mg * self.mg * 512))
                layers.append(batch_norm()(layers[-1], train=train))
                layers.append(tf.nn.relu(layers[-1]))
                layers.append(tf.reshape(
                    layers[-1], [-1, self.mg, self.mg, 512]))

            with tf.variable_scope("layer1"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, self.mg * 2, self.mg * 2, 256],
                    d_h=2,
                    d_w=2,
                    k_h=4,
                    k_w=4))
                layers.append(batch_norm()(layers[-1], train=train))
                layers.append(tf.nn.relu(layers[-1]))

            with tf.variable_scope("layer2"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, self.mg * 4, self.mg * 4, 128],
                    d_h=2,
                    d_w=2,
                    k_h=4,
                    k_w=4))
                layers.append(batch_norm()(layers[-1], train=train))
                layers.append(tf.nn.relu(layers[-1]))

            with tf.variable_scope("layer3"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, self.mg * 8, self.mg * 8, 64],
                    d_h=2,
                    d_w=2,
                    k_h=4,
                    k_w=4))
                layers.append(batch_norm()(layers[-1], train=train))
                layers.append(tf.nn.relu(layers[-1]))

            with tf.variable_scope("layer4"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, self.output_height,
                     self.output_width, self.output_depth],
                    d_h=1,
                    d_w=1,
                    k_h=3,
                    k_w=3))
                layers.append(tf.nn.tanh(layers[-1]))

            return layers[-1], layers


class Discriminator(Network):
    def __init__(self, sn_mode, scale_initial=1.0, scale_trainable=True,
                 scope_name="discriminator", *args, **kwargs):
        super(Discriminator, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self.sn_mode = sn_mode
        self.scale_initial = scale_initial
        self.scale_trainable = scale_trainable

        self.NUM_LAYERS = 8

    def build(self, images, train, sn_op):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            scales = []
            for i in range(self.NUM_LAYERS):
                scale = tf.get_variable(
                    "scale{}".format(i),
                    shape=[],
                    initializer=tf.constant_initializer(self.scale_initial),
                    trainable=self.scale_trainable)
                scales.append(scale)

            layers = [images]
            sigmas = []
            with tf.variable_scope("layer0"):
                s_layer, s_sigma = conv2d(
                    layers[-1],
                    64,
                    d_h=1,
                    d_w=1,
                    k_h=3,
                    k_w=3,
                    sn_op=sn_op,
                    with_sigma=True,
                    scale=scales[0],
                    sn_mode=self.sn_mode)
                layers.append(s_layer)
                sigmas.append(s_sigma)

                layers.append(lrelu(layers[-1], leak=0.1))

            with tf.variable_scope("layer1"):
                s_layer, s_sigma = conv2d(
                    layers[-1],
                    64,
                    d_h=2,
                    d_w=2,
                    k_h=4,
                    k_w=4,
                    sn_op=sn_op,
                    with_sigma=True,
                    scale=scales[1],
                    sn_mode=self.sn_mode)
                layers.append(s_layer)
                sigmas.append(s_sigma)

                layers.append(lrelu(layers[-1], leak=0.1))

            with tf.variable_scope("layer2"):
                s_layer, s_sigma = conv2d(
                    layers[-1],
                    128,
                    d_h=1,
                    d_w=1,
                    k_h=3,
                    k_w=3,
                    sn_op=sn_op,
                    with_sigma=True,
                    scale=scales[2],
                    sn_mode=self.sn_mode)
                layers.append(s_layer)
                sigmas.append(s_sigma)

                layers.append(lrelu(layers[-1], leak=0.1))

            with tf.variable_scope("layer3"):
                s_layer, s_sigma = conv2d(
                    layers[-1],
                    128,
                    d_h=2,
                    d_w=2,
                    k_h=4,
                    k_w=4,
                    sn_op=sn_op,
                    with_sigma=True,
                    scale=scales[3],
                    sn_mode=self.sn_mode)
                layers.append(s_layer)
                sigmas.append(s_sigma)

                layers.append(lrelu(layers[-1], leak=0.1))

            with tf.variable_scope("layer4"):
                s_layer, s_sigma = conv2d(
                    layers[-1],
                    256,
                    d_h=1,
                    d_w=1,
                    k_h=3,
                    k_w=3,
                    sn_op=sn_op,
                    with_sigma=True,
                    scale=scales[4],
                    sn_mode=self.sn_mode)
                layers.append(s_layer)
                sigmas.append(s_sigma)

                layers.append(lrelu(layers[-1], leak=0.1))

            with tf.variable_scope("layer5"):
                s_layer, s_sigma = conv2d(
                    layers[-1],
                    256,
                    d_h=2,
                    d_w=2,
                    k_h=4,
                    k_w=4,
                    sn_op=sn_op,
                    with_sigma=True,
                    scale=scales[5],
                    sn_mode=self.sn_mode)
                layers.append(s_layer)
                sigmas.append(s_sigma)

                layers.append(lrelu(layers[-1], leak=0.1))

            with tf.variable_scope("layer6"):
                s_layer, s_sigma = conv2d(
                    layers[-1],
                    512,
                    d_h=1,
                    d_w=1,
                    k_h=3,
                    k_w=3,
                    sn_op=sn_op,
                    with_sigma=True,
                    scale=scales[6],
                    sn_mode=self.sn_mode)
                layers.append(s_layer)
                sigmas.append(s_sigma)

                layers.append(lrelu(layers[-1], leak=0.1))

            with tf.variable_scope("layer7"):
                s_layer, s_sigma = linear(
                    layers[-1],
                    1,
                    sn_op=sn_op,
                    with_sigma=True,
                    scale=scales[7])
                layers.append(s_layer)
                sigmas.append(s_sigma)

            return layers[-1], layers, sigmas, scales
