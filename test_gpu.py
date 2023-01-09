"""
Copyright (c) 2022, David Lee
Author: David Lee
"""
import sys
import os
import logging
from ncnn_vulkan import ncnn
import argparse
from upscale.upscale_processing import upscale_image
import time


def test_gpu(gpu=None):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    gpu_count = ncnn.get_gpu_count()

    logging.info("Searching for Vulkan compatible GPUs")
    logging.info("====================================")
    logging.info("GPU count: " + str(gpu_count))
    logging.info("====================================")
    logging.info("Default GPU: " + str(ncnn.get_default_gpu_index()))
    logging.info("====================================")

    gpu_types = ["Discrete", "Integrated", "Virtual", "CPU"]

    for i in range(gpu_count):
        gpu_info = ncnn.get_gpu_info(i)
        logging.info(
            "GPU "
            + str(i)
            + ": "
            + gpu_types[gpu_info.type()]
            + " / "
            + gpu_info.device_name()
        )

    net = ncnn.Net()

    # test vulkan compute for gpu
    if gpu is not None:
        logging.info("====================================")
        logging.info("Testing GPU " + str(gpu) + " - Check for error messages")

        if gpu + 1 > gpu_count:
            sys.exit("GPU " + str(gpu) + " is not a valid GPU!")

        if gpu >= 0:
            net.opt.use_vulkan_compute = True
            net.set_vulkan_device(gpu)

        current_path = os.path.realpath(__file__).split(os.sep)[:-1]
        model_path = os.sep.join(current_path + ["models"])
        input_file = os.sep.join(current_path + ["sample.png"])

        scale = 2

        net.load_param(
            os.path.join(model_path, str(scale) + "x_Compact_Pretrain.param")
        )
        net.load_model(os.path.join(model_path, str(scale) + "x_Compact_Pretrain.bin"))
        input_name = "input"
        output_name = "output"

        start = time.time()
        upscale_image(input_file, None, scale, net, input_name, output_name)
        total = time.time() - start
        logging.info(str(total) + " seconds to upscale sample.png")
        logging.info("Testing GPU " + str(gpu) + " - Passed")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test GPU - List GPUs")
    parser.add_argument(
        "-g", "--gpu", type=int, help="Optional gpu number to test. -1 to test cpu."
    )
    args = parser.parse_args()

    test_gpu(args.gpu)
