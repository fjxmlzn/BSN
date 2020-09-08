import tensorflow as tf
from op import linear, batch_norm, conv2d, up_resnet, down_resnet
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

    def build(self, z, train, var_device):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            layers = [z]

            with tf.variable_scope("layer0"):
                layers.append(linear(
                    layers[-1],
                    self.mg * self.mg * 1024,
                    var_device=var_device,
                    ini_scale=1.0))
                layers.append(tf.reshape(
                    layers[-1], [-1, self.mg, self.mg, 1024]))

            with tf.variable_scope("layer1"):
                layers.append(up_resnet(
                    layers[-1],
                    1024,
                    1024,
                    train=train,
                    var_device=var_device))

            with tf.variable_scope("layer2"):
                layers.append(up_resnet(
                    layers[-1],
                    512,
                    512,
                    train=train,
                    var_device=var_device))

            with tf.variable_scope("layer3"):
                layers.append(up_resnet(
                    layers[-1],
                    256,
                    256,
                    train=train,
                    var_device=var_device))

            with tf.variable_scope("layer4"):
                layers.append(up_resnet(
                    layers[-1],
                    128,
                    128,
                    train=train,
                    var_device=var_device))

            with tf.variable_scope("layer5"):
                layers.append(up_resnet(
                    layers[-1],
                    64,
                    64,
                    train=train,
                    var_device=var_device))

            with tf.variable_scope("layer6"):
                layers.append(batch_norm()(layers[-1], train=train))
                layers.append(tf.nn.relu(layers[-1]))
                layers.append(conv2d(
                    layers[-1],
                    self.output_depth,
                    d_h=1,
                    d_w=1,
                    k_h=3,
                    k_w=3,
                    sn_op=None,
                    with_sigma=False,
                    var_device=var_device,
                    ini_scale=1.0))
                layers.append(tf.nn.tanh(layers[-1]))

            return layers[-1], layers


class Discriminator(Network):
    def __init__(self, scale, sn_mode,
                 scope_name="discriminator", *args, **kwargs):
        super(Discriminator, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self.scale = scale
        self.sn_mode = sn_mode

    def build(self, images, train, sn_op, var_device):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            layers = [images]
            sigmas = []
            with tf.variable_scope("layer0"):
                s_layer, s_sigma = down_resnet(
                    layers[-1],
                    64,
                    64,
                    train=train,
                    sn_op=sn_op,
                    with_sigma=True,
                    var_device=var_device,
                    first=True,
                    scale=self.scale,
                    sn_mode=self.sn_mode)
                layers.append(s_layer)
                sigmas.append(s_sigma)

            with tf.variable_scope("layer1"):
                s_layer, s_sigma = down_resnet(
                    layers[-1],
                    128,
                    128,
                    train=train,
                    sn_op=sn_op,
                    with_sigma=True,
                    var_device=var_device,
                    scale=self.scale,
                    sn_mode=self.sn_mode)
                layers.append(s_layer)
                sigmas.append(s_sigma)

            with tf.variable_scope("layer2"):
                s_layer, s_sigma = down_resnet(
                    layers[-1],
                    256,
                    256,
                    train=train,
                    sn_op=sn_op,
                    with_sigma=True,
                    var_device=var_device,
                    scale=self.scale,
                    sn_mode=self.sn_mode)
                layers.append(s_layer)
                sigmas.append(s_sigma)

            with tf.variable_scope("layer3"):
                s_layer, s_sigma = down_resnet(
                    layers[-1],
                    512,
                    512,
                    train=train,
                    sn_op=sn_op,
                    with_sigma=True,
                    var_device=var_device,
                    scale=self.scale,
                    sn_mode=self.sn_mode)
                layers.append(s_layer)
                sigmas.append(s_sigma)

            with tf.variable_scope("layer4"):
                s_layer, s_sigma = down_resnet(
                    layers[-1],
                    1024,
                    1024,
                    train=train,
                    sn_op=sn_op,
                    with_sigma=True,
                    var_device=var_device,
                    scale=self.scale,
                    sn_mode=self.sn_mode)
                layers.append(s_layer)
                sigmas.append(s_sigma)

            with tf.variable_scope("layer5"):
                s_layer, s_sigma = down_resnet(
                    layers[-1],
                    1024,
                    1024,
                    train=train,
                    sn_op=sn_op,
                    with_sigma=True,
                    down=False,
                    var_device=var_device,
                    scale=self.scale,
                    sn_mode=self.sn_mode)
                layers.append(s_layer)
                sigmas.append(s_sigma)

            with tf.variable_scope("layer6"):
                layers.append(tf.nn.relu(layers[-1]))
                layers.append(
                    tf.reduce_sum(layers[-1], reduction_indices=[1, 2]))
                s_layer, s_sigma = linear(
                    layers[-1],
                    1,
                    sn_op=sn_op,
                    with_sigma=True,
                    var_device=var_device,
                    ini_scale=1.0,
                    scale=self.scale)
                layers.append(s_layer)
                sigmas.append(s_sigma)

            return layers[-1], layers, sigmas
