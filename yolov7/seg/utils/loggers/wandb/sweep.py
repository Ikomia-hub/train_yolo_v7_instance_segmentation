import sys
from pathlib import Path

import wandb

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv5 root directory


from train_yolo_v7_instance_segmentation.yolov7.seg.segment.train import parse_opt, train
from train_yolo_v7_instance_segmentation.yolov7.seg.utils.callbacks import Callbacks
from train_yolo_v7_instance_segmentation.yolov7.seg.utils.general import increment_path
from train_yolo_v7_instance_segmentation.yolov7.seg.utils.torch_utils import select_device


def sweep():
    wandb.init()
    # Get hyp dict from sweep agent. Copy because train() modifies parameters which confused wandb.
    hyp_dict = vars(wandb.config).get("_items").copy()

    # Workaround: get necessary opt args
    opt = parse_opt(known=True)
    opt.batch_size = hyp_dict.get("batch_size")
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.evolve))
    opt.epochs = hyp_dict.get("epochs")
    opt.nosave = True
    opt.data = hyp_dict.get("data")
    opt.weights = str(opt.weights)
    opt.cfg = str(opt.cfg)
    opt.data = str(opt.data)
    opt.hyp = str(opt.hyp)
    opt.project = str(opt.project)
    device = select_device(opt.device, batch_size=opt.batch_size)

    # train
    train(hyp_dict, opt, device, callbacks=Callbacks())


if __name__ == "__main__":
    sweep()
