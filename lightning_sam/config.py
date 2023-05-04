from box import Box
import os

try:
    training_dir = os.environ["SAM_TRAINING_DIR"]
    training_coco = os.environ["SAM_TRAINING_COCO"]

    valid_dir = os.environ["SAM_VALID_DIR"]
    valid_coco = os.environ["SAM_VALID_COCO"]

    sam_chkpt = os.environ["SAM_CHECKPOINT"]
except KeyError as err:
    raise ValueError(f"Set {err} environment variable with absolute path")

config = {
    "num_devices": 4,
    "batch_size": 12,
    "grad_update": 1,
    "num_workers": 4,
    "num_epochs": 20,
    "eval_interval": 2,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_h',
        "checkpoint": sam_chkpt,
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": training_dir,
            "annotation_file": training_coco
        },
        "val": {
            "root_dir": valid_dir,
            "annotation_file": valid_coco
        }
    }
}

cfg = Box(config)
