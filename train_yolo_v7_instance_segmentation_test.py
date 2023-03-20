import logging
from ikomia.core import task
from ikomia.utils.tests import run_for_test

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::train yolo v7 instance segmentation =====")
    input_dataset = t.get_input(0)
    params = task.get_parameters(t)
    params["epochs"] = 2
    params["batch_size"] = 1
    params["dataset_split_ratio"] = 0.5
    task.set_parameters(t, params)
    input_dataset.load(data_dict["datasets"]["instance_segmentation"]["dataset_coco"])
    yield run_for_test(t)