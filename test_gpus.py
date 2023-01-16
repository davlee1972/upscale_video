"""
Copyright (c) 2022, David Lee
Author: David Lee
"""
import sys
import os
import logging
from ncnn_vulkan import ncnn
import argparse
from upscale.upscale_processing import upscale_image, init_worker, logging_callback
import time
import multiprocessing


def upscale_images(input_file_name, output_file_name, scale, gpus):

    logging_items = []

    i = int(multiprocessing.current_process()._identity[0]) - 1

    start = time.time()

    logging_items.append(["info", "Testing GPU: " + str(gpus[i])])

    ret = upscale_image(
        input_file_name, output_file_name, scale, None, 1, 1, remove=False
    )

    logging_items + ret

    total = time.time() - start

    logging_items.append(["info", str(total) + " seconds to upscale sample.png"])

    return logging_items


def run_tests(gpus=None, scale=None, runs=None):

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

    # test vulkan compute for gpu
    if gpus is not None:

        if gpus:
            gpus = gpus.split(",")
            gpus = [int(g) for g in gpus]

        current_path = os.path.realpath(__file__).split(os.sep)[:-1]
        model_path = os.sep.join(current_path + ["models"])

        pool = multiprocessing.get_context("spawn").Pool(
            processes=len(gpus),
            initializer=init_worker,
            initargs=(
                gpus,
                0,
                model_path,
                "x_Compact_Pretrain",
                scale,
                "input",
                "output",
            ),
        )

        logging.info("")
        logging.info("Starting test runs")
        logging.info("====================================")
        start = time.time()

        for i in range(runs):
            input_file = os.sep.join(current_path + ["sample.png"])
            pool.apply_async(
                upscale_images,
                args=(input_file, None, scale, gpus),
                callback=logging_callback,
            )

        pool.close()
        pool.join()

        total = time.time() - start

        logging.info("====================================")
        logging.info(str(total) + " seconds total to run tests.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test GPU - List GPUs")
    parser.add_argument(
        "-g", "--gpus", help="Optional gpus to test. Example 0,1,1,2. Default is 0."
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=2, help="Scale 2 or 4. Default is 2."
    )
    parser.add_argument("-r", "--runs", type=int, default=10, help="Number of tests")
    args = parser.parse_args()

    run_tests(args.gpus, args.scale, args.runs)
