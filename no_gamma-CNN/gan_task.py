from gpu_task_scheduler.gpu_task import GPUTask


class GANTask(GPUTask):
    def main(self):
        import os
        import tensorflow as tf
        from gan.load_data import load_celeba, load_stl10, load_cifar10
        from gan.latent import GaussianLatent
        from gan.network import Generator, Discriminator
        from gan.gan import GAN
        from gan.metric import InceptionScore, FrechetInceptionDistance

        if self._config["dataset"] == "celeba":
            data = load_celeba("data/CelebA")
        elif self._config["dataset"] == "stl10":
            data = load_stl10("data/STL10")
        elif self._config["dataset"] == "cifar10":
            data = load_cifar10("data/CIFAR10")
        else:
            raise ValueError("Unknown dataset: {}".format(
                self._config["dataset"]))

        _, height, width, depth = data.shape
        print(data.min(), data.max())

        latent = GaussianLatent(
            dim=self._config["latent_dim"], loc=0.0, scale=1.0)

        generator = Generator(
            output_width=width,
            output_height=height,
            output_depth=depth,
            mg=self._config["mg"])
        discriminator = Discriminator(
            scale=self._config["scale"], sn_mode=self._config["sn_mode"])

        checkpoint_dir = os.path.join(self._work_dir, "checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        sample_dir = os.path.join(self._work_dir, "sample")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        time_path = os.path.join(self._work_dir, "time.txt")
        metric_path = os.path.join(self._work_dir, "metrics.csv")

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
                iteration=self._config["iteration"],
                batch_size=self._config["batch_size"],
                data=data,
                vis_freq=self._config["vis_freq"],
                vis_num_h=self._config["vis_num_h"],
                vis_num_w=self._config["vis_num_w"],
                latent=latent,
                generator=generator,
                discriminator=discriminator,
                n_dis=self._config["n_dis"],
                metric_callbacks=metric_callbacks,
                metric_freq=self._config["metric_freq"],
                metric_path=metric_path,
                g_lr=self._config["g_lr"],
                g_beta1=self._config["g_beta1"],
                g_beta2=self._config["g_beta2"],
                d_lr=self._config["d_lr"],
                d_beta1=self._config["d_beta1"],
                d_beta2=self._config["d_beta2"],
                summary_freq=self._config["summary_freq"],
                save_checkpoint_freq=self._config["save_checkpoint_freq"],
                extra_checkpoint_freq=self._config["extra_checkpoint_freq"])
            gan.build()

            if "restore" in self._config and self._config["restore"]:
                gan.train(restore=True)
            else:
                gan.train()
