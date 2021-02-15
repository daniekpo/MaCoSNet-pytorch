def eval_args():
    args = {
        "batch_size": 16,
        "eval_dataset": 'pf-pascal',
        "eval_metric": 'pck',
        "feature_extraction_cnn": 'resnet101',
        "feature_extraction_last_layer": '',
        "flow_output_dir": 'results/',
        "fr_channels": [128, 64],
        "fr_feature_size": 15,
        "fr_kernel_sizes": [7, 5],
        "gpu": 0,
        "image_size": 240,
        "log_dir": '',
        "model": '',
        "model_aff": '',
        "model_tps": '',
        "model_type": 'match',
        "num_workers": 8,
        "pck_alpha": 0.1,
        "self_correlation": False,
        "tps_reg_factor": 0.0
    }
    arg_groups = {
        "positional arguments": {},
        "optional arguments": {"help": None},
        "base": {
            "image_size": 240,
            "model": "",
            "model_aff": "",
            "model_tps": "",
            "model_type": "match",
            "gpu": 0,
            "num_workers": 8,
        },
        "model": {
            "feature_extraction_cnn": "resnet101",
            "feature_extraction_last_layer": "",
            "fr_feature_size": 15,
            "fr_kernel_sizes": [7, 5],
            "fr_channels": [128, 64],
        },
        "eval": {
            "eval_dataset": "pf-pascal",
            "flow_output_dir": "results/",
            "pck_alpha": 0.1,
            "eval_metric": "pck",
            "tps_reg_factor": 0.0,
            "batch_size": 16,
            "log_dir": "",
            "self_correlation": False,
        }
    }

    return args, arg_groups
