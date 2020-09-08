config = {
    "scheduler_config": {
        "gpu": ["0"],
        "temp_folder": "temp",
        "scheduler_log_file_path": "scheduler.log",
        "log_file": "worker.log",
        #"force_rerun": True
    },

    "global_config": {
        "iteration": 500000,
        "vis_freq": 200,
        "vis_num_h": 10,
        "vis_num_w": 10,
        "latent_dim": 128,
        "summary_freq": 1,
        "metric_freq": 10000,
        "extra_checkpoint_freq": 50000,
        "save_checkpoint_freq": 10000,
        #"restore": True
    },

    "test_config": [
        {
            "run": range(3),
            "g_lr": [0.0002],
            "d_lr": [0.0002],
            "g_beta1": [0.0],
            "d_beta1": [0.0],
            "g_beta2": [0.9],
            "d_beta2": [0.9],
            "n_dis": [5],
            "num_gpus": [1],
            "d_batch_size": [16],
            "g_batch_size": [32],
            "scale": [1.0, 1.2, 1.4, 1.6],
            "sn_mode": ["SN", "BSN"]
        }
    ]
}
