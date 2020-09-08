config = {
    "scheduler_config": {
        "gpu": ["0"],
        "temp_folder": "temp",
        "scheduler_log_file_path": "scheduler.log",
        "log_file": "worker.log",
        #"force_rerun": True
    },

    "global_config": {
        "batch_size": 64,
        "vis_freq": 200,
        "vis_num_h": 10,
        "vis_num_w": 10,
        "latent_dim": 128,
        "summary_freq": 1,
        "metric_freq": 10000,
        "extra_checkpoint_freq": 100000,
        "save_checkpoint_freq": 10000,
        "d_gp_coe": 10.0,
        #"restore": True
    },

    "test_config": [
        {
            "dataset": ["cifar10"],
            "iteration": [400000],
            "run": range(5),
            "g_lr": [0.0001, 0.0002],
            "d_lr": [0.0001, 0.0002],
            "g_beta1": [0.5],
            "d_beta1": [0.5],
            "g_beta2": [0.999],
            "d_beta2": [0.999],
            "n_dis": [1],
            "mg": [4],
            "sn_mode": ["SN"]
        },
        {
            "dataset": ["stl10"],
            "iteration": [800000],
            "run": range(5),
            "g_lr": [0.0001, 0.0002],
            "d_lr": [0.0001, 0.0002],
            "g_beta1": [0.5],
            "d_beta1": [0.5],
            "g_beta2": [0.999],
            "d_beta2": [0.999],
            "n_dis": [1],
            "mg": [6],
            "sn_mode": ["SN"]
        },
        {
            "dataset": ["celeba"],
            "iteration": [400000],
            "run": range(5),
            "g_lr": [0.0001, 0.0002],
            "d_lr": [0.0001, 0.0002],
            "g_beta1": [0.5],
            "d_beta1": [0.5],
            "g_beta2": [0.999],
            "d_beta2": [0.999],
            "n_dis": [1],
            "mg": [8],
            "sn_mode": ["SN"]
        },
        {
            "dataset": ["cifar10"],
            "iteration": [400000],
            "run": range(5),
            "g_lr": [0.0001],
            "d_lr": [0.0001],
            "g_beta1": [0.5],
            "d_beta1": [0.5],
            "g_beta2": [0.9],
            "d_beta2": [0.9],
            "n_dis": [5],
            "mg": [4],
            "sn_mode": ["SN"]
        },
        {
            "dataset": ["stl10"],
            "iteration": [400000],
            "run": range(5),
            "g_lr": [0.0001],
            "d_lr": [0.0001],
            "g_beta1": [0.5],
            "d_beta1": [0.5],
            "g_beta2": [0.9],
            "d_beta2": [0.9],
            "n_dis": [5],
            "mg": [6],
            "sn_mode": ["SN"]
        },
        {
            "dataset": ["celeba"],
            "iteration": [400000],
            "run": range(5),
            "g_lr": [0.0001],
            "d_lr": [0.0001],
            "g_beta1": [0.5],
            "d_beta1": [0.5],
            "g_beta2": [0.9],
            "d_beta2": [0.9],
            "n_dis": [5],
            "mg": [8],
            "sn_mode": ["SN"]
        }
    ]
}
