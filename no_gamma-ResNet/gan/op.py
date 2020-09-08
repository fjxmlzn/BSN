import tensorflow as tf
import numpy as np
from sn import spectral_normed_weight


def linear(input_, output_size, scope_name="linear",
           ini_scale=1.0, bias_start=0.0, sn_op=None, with_sigma=False,
           var_device="/cpu:0", scale=None):
    with tf.variable_scope(scope_name):
        input_ = tf.reshape(
            input_,
            [-1, np.prod(input_.get_shape().as_list()[1:])])
        # output = tf.layers.dense(
        #    input_,
        #    output_size)
        with tf.device(var_device):
            matrix = tf.get_variable(
                "matrix",
                [input_.get_shape().as_list()[1], output_size],
                tf.float32,
                tf.initializers.variance_scaling(
                    scale=ini_scale,
                    mode="fan_avg",
                    distribution="uniform"))
        if sn_op is not None:
            matrix, sigma = spectral_normed_weight(
                matrix,
                update_collection=sn_op,
                with_sigma=True,
                var_device=var_device)
        else:
            sigma = None

        if scale is not None:
            matrix = matrix * scale
            
        with tf.device(var_device):
            bias = tf.get_variable(
                "bias",
                [output_size],
                initializer=tf.constant_initializer(bias_start))

        output = tf.matmul(input_, matrix) + bias

        if with_sigma:
            return output, sigma
        else:
            return output


def flatten(input_, scope_name="flatten"):
    with tf.variable_scope(scope_name):
        output = tf.reshape(
            input_,
            [-1, np.prod(input_.get_shape().as_list()[1:])])
        return output


class batch_norm(object):
    # Code from:
    # https://github.com/carpedm20/DCGAN-tensorflow
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            #updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, ini_scale=1.0,
             name="deconv2d",
             var_device="/cpu:0"):
    # Code modified from:
    # https://github.com/carpedm20/DCGAN-tensorflow
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        with tf.device(var_device):
            w = tf.get_variable(
                'w',
                [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.initializers.variance_scaling(
                    scale=ini_scale,
                    mode="fan_avg",
                    distribution="uniform"))

        try:
            deconv = tf.nn.conv2d_transpose(
                input_,
                w,
                output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(
                input_,
                w,
                output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

        with tf.device(var_device):
            biases = tf.get_variable(
                'biases',
                [output_shape[-1]],
                initializer=tf.constant_initializer(0.0))

        deconv = tf.reshape(
            tf.nn.bias_add(deconv, biases), output_shape)

        return deconv


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, ini_scale=1.0,
           name="conv2d", sn_op=None, with_sigma=False,
           var_device="/cpu:0", scale=None, sn_mode=None):
    # Code from:
    # https://github.com/carpedm20/DCGAN-tensorflow
    with tf.variable_scope(name):
        shape = input_.get_shape().as_list()
        assert shape[1] == shape[2]
        with tf.device(var_device):
            w = tf.get_variable(
                'w',
                [k_h, k_w, input_.get_shape()[-1], output_dim],
                initializer=tf.initializers.variance_scaling(
                    scale=ini_scale,
                    mode="fan_avg",
                    distribution="uniform"))
        if sn_op is not None:
            if sn_mode == "BSN":
                with tf.variable_scope("SN1"):
                    _, sigma1 = spectral_normed_weight(
                        w,
                        update_collection=sn_op,
                        with_sigma=True)
                with tf.variable_scope("SN2"):
                    _, sigma2 = spectral_normed_weight(
                        w,
                        update_collection=sn_op,
                        with_sigma=True,
                        transpose=True)
                sigma = (sigma1 + sigma2) / 2
                w = w / sigma
                sigmas = [sigma, sigma1, sigma2]
            elif sn_mode == "SN":
                w, sigma = spectral_normed_weight(
                    w,
                    update_collection=sn_op,
                    with_sigma=True)
                sigmas = sigma
            else:
                raise ValueError("Unknown SN mode: {}".format(sn_mode))
        else:
            sigmas = None

        if scale is not None:
            w = w * scale

        conv = tf.nn.conv2d(
            input_,
            w,
            strides=[1, d_h, d_w, 1],
            padding='SAME')
        with tf.device(var_device):
            biases = tf.get_variable(
                'biases',
                [output_dim],
                initializer=tf.constant_initializer(0.0))

        conv = tf.reshape(
            tf.nn.bias_add(conv, biases),
            [-1] + conv.get_shape().as_list()[1:])

        if with_sigma:
            return conv, sigmas
        else:
            return conv


def lrelu(x, leak=0.2, name="lrelu"):
    # Code from:
    # https://github.com/carpedm20/DCGAN-tensorflow
    return tf.maximum(x, leak * x)

def unpooling(input_, k_size, name="unpooling"):
    batch_size = tf.shape(input_)[0]

    h = input_.get_shape().as_list()[1]
    w = input_.get_shape().as_list()[2]
    c = input_.get_shape().as_list()[3]

    # compute deconv kernel
    kernel1 = np.ones(shape=(k_size, k_size, 1), dtype=np.float32)
    kernel2 = np.eye(c)
    kernel2 = np.expand_dims(kernel2, axis=0)
    kernel = np.einsum("ijk,klm", kernel1, kernel2)

    # compute deconv
    with tf.variable_scope(name):
        deconv = tf.nn.conv2d_transpose(
            input_,
            tf.constant(kernel, dtype=tf.float32),
            output_shape=(batch_size, h * k_size, w * k_size, c),
            strides=[1, k_size, k_size, 1])
        return deconv

def up_resnet(input_, hidden_dim, output_dim, train,
              k_h=3, k_w=3, scope_name="up_resnet",
              var_device="/cpu:0"):
    with tf.variable_scope(scope_name):
        output = input_

        with tf.variable_scope("block1"):
            output = batch_norm()(output, train=train)
            output = tf.nn.relu(output)
            output = unpooling(output, 2)
            output = conv2d(
                output,
                hidden_dim,
                d_h=1,
                d_w=1,
                k_h=k_h,
                k_w=k_w,
                sn_op=None,
                with_sigma=False,
                var_device=var_device,
                ini_scale=2.0)
        with tf.variable_scope("block2"):
            output = batch_norm()(output, train=train)
            output = tf.nn.relu(output)
            output = conv2d(
                output,
                output_dim,
                d_h=1,
                d_w=1,
                k_h=k_h,
                k_w=k_w,
                sn_op=None,
                with_sigma=False,
                var_device=var_device,
                ini_scale=2.0)
        with tf.variable_scope("shortcut"):
            output2 = unpooling(input_, 2)
            output2 = conv2d(
                output2,
                output_dim,
                d_h=1,
                d_w=1,
                k_h=1,
                k_w=1,
                sn_op=None,
                with_sigma=False,
                var_device=var_device,
                ini_scale=1.0)
        return output + output2


def down_resnet(input_, hidden_dim, output_dim, train, down=True,
                k_h=3, k_w=3, scope_name="down_resnet",
                sn_op=None, with_sigma=False, var_device="/cpu:0",
                first=False, scale=1.0, sn_mode=None):
    with tf.variable_scope(scope_name):
        output = input_
        
        sigmas = []
        with tf.variable_scope("block1"):
            if not first:
                output = tf.nn.relu(output)
            output, sigma = conv2d(
                output,
                hidden_dim,
                d_h=1,
                d_w=1,
                k_h=k_h,
                k_w=k_w,
                sn_op=sn_op,
                with_sigma=True,
                var_device=var_device,
                ini_scale=2.0,
                scale=scale,
                sn_mode=sn_mode)
            sigmas.append(sigma)
        with tf.variable_scope("block2"):
            output = tf.nn.relu(output)
            output, sigma = conv2d(
                output,
                output_dim,
                d_h=1,
                d_w=1,
                k_h=k_h,
                k_w=k_w,
                sn_op=sn_op,
                with_sigma=True,
                var_device=var_device,
                ini_scale=2.0,
                scale=scale,
                sn_mode=sn_mode)
            sigmas.append(sigma)
            if down:
                output = tf.nn.avg_pool(
                    output, ksize=2, strides=2, padding="SAME")
        with tf.variable_scope("shortcut"):
            output2 = input_
            if not first:
                output2, sigma = conv2d(
                    output2,
                    output_dim,
                    d_h=1,
                    d_w=1,
                    k_h=1,
                    k_w=1,
                    sn_op=sn_op,
                    with_sigma=True,
                    var_device=var_device,
                    ini_scale=1.0,
                    scale=scale,
                    sn_mode=sn_mode)
                sigmas.append(sigma)
                if down:
                    output2 = tf.nn.avg_pool(
                        output2, ksize=2, strides=2, padding="SAME")
            else:
                if down:
                    output2 = tf.nn.avg_pool(
                        output2, ksize=2, strides=2, padding="SAME")
                output2, sigma = conv2d(
                    output2,
                    output_dim,
                    d_h=1,
                    d_w=1,
                    k_h=1,
                    k_w=1,
                    sn_op=sn_op,
                    with_sigma=True,
                    var_device=var_device,
                    ini_scale=1.0,
                    scale=scale,
                    sn_mode=sn_mode)
                sigmas.append(sigma)

        final_output = output + output2
        if with_sigma:
            return final_output, sigmas
        else:
            return final_output
