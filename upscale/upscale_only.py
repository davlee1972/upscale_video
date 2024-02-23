"""
Copyright (c) 2022, David Lee
Author: David Lee
"""

import argparse
import logging
import os
import tempfile
import subprocess
import sys
import zipfile
import shutil


from upscale_processing import (
    get_metadata,
    get_crop_detect,
    calc_batches,
    extract_frames,
    process_model,
    process_denoise,
    upscale_frames,
)


def upscale_only(
    input_file,
    ffmpeg,
    scale,
    temp_dir,
    batch_size,
    gpus,
    upscale_dir,
    extract_only,
    models,
    log_level,
    log_dir,
):
    """
    Upscales video file 2x or 4x only

    :param input_file:
    :param ffmpeg:
    :param scale:
    :param temp_dir:
    :param batch_size:
    :param gpus:
    :param extract_only:
    :param models:
    :param log_level:
    :param log_dir:
    """

    if scale not in [1, 2, 4]:
        sys.exit("Scale must be 2 or 4 - Exiting")

    if not os.path.exists(input_file):
        sys.exit(input_file + " not found")

    if upscale_dir and not os.path.exists(upscale_dir):
        sys.exit(upscale_dir + " is not not valid")

    if models:
        models = models.split(",")
    else:
        models = []

    if "r" in models:
        scale = 4

    denoise = [model.split("=") for model in models if model.startswith("n=")]

    if denoise:
        denoise = int(denoise[0][1])
        if denoise > 30:
            denoise = 30
        if denoise <= 0:
            denoise = None

    if not log_level:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    if log_dir:
        log_file = os.path.join(log_dir, input_file.split(os.sep)[-1][:-4] + ".log")
        # create log file handler and set level to debug
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        fh.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(fh)

    if gpus:
        try:
            gpus = gpus.split(",")
            gpus = [int(g) for g in gpus]
        except ValueError:
            logging.error("Invalid gpus")
            sys.exit("Error - Exiting")
    else:
        gpus = [0]

    logging.info("Processing File: " + input_file)

    ## Create temp directory
    if not temp_dir:
        temp_dir = tempfile.gettempdir()

    temp_dir = os.path.abspath(os.path.join(temp_dir, "upscale_video"))
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    ## change working directory to temp directory
    os.chdir(temp_dir)

    if os.path.exists("upscaled.txt"):
        sys.exit(input_file + " already processed - Exiting")

    with keep.running() as m:

        ## get metadata
        info_dict = get_metadata(ffmpeg, input_file)

        frames_count = info_dict["number_of_frames"]
        frame_rate = info_dict["frame_rate"]
        duration = info_dict["duration"]

        ## calculate frames per minute
        ## calculate frames per minute and batches
        if batch_size > 0:
            frames_per_batch = int(frame_rate * 60) * batch_size
        else:
            frames_per_batch = int(frames_count / (-1 * batch_size)) + 100

        frame_batches = calc_batches(frames_count, frames_per_batch)

        crop_detect = get_crop_detect(ffmpeg, input_file, duration)

        extract_frames(
            ffmpeg,
            input_file,
            crop_detect,
            info_dict,
            frames_count,
            {1: [1, frames_count]},
            extract_only,
        )

        model_path = os.path.realpath(__file__).split(os.sep)
        model_path = os.sep.join(model_path[:-2] + ["models"])

        workers_used = 0
        input_file_tag = "extract"

        if denoise:
            logging.info("Starting denoise touchup...")
            workers_used += process_denoise(frames_count, input_file_tag, denoise)
            input_file_tag = "denoise"

        if "a" in models:
            logging.info("Starting anime touchup...")

            model_file = "x_HurrDeblur_SubCompact_nf24-nc8_244k_net_g"
            output_file_tag = "anime"

            process_model(
                frames_count,
                model_path,
                model_file,
                1,
                "input",
                "output",
                input_file_tag,
                output_file_tag,
                gpus,
                workers_used,
            )

            workers_used += len(gpus)
            input_file_tag = "anime"

        if upscale_dir:
            shutil.copyfile("metadata.json", os.path.join(upscale_dir, "metadata.json"))
            shutil.copyfile("crop_detect.txt", os.path.join(upscale_dir, "crop_detect.txt"))

        logging.info("Starting upscale processing...")

        if "r" in models:
            model_file = "x_Valar_v1"
            model_input = "input"
            model_output = "output"
        else:
            model_file = "x_Compact_Pretrain"
            model_input = "input"
            model_output = "output"

        ## process input file in batches
        for frame_batch, frame_range in frame_batches.items():

            if upscale_dir:
                if os.path.exists(os.path.join(upscale_dir, str(frame_batch) + ".zip")):
                    continue
            else:
                if os.path.exists(str(frame_batch) + ".zip"):
                    continue

            if scale == 1:
                for frame in range(frame_range[0], frame_range[1] + 1):
                    os.rename(
                        str(frame) + "." + input_file_tag + ".png", str(frame) + ".png"
                    )
            else:
                upscale_frames(
                    frame_batch,
                    frame_range[0],
                    frame_range[1],
                    input_file_tag,
                    scale,
                    gpus,
                    workers_used,
                    model_path,
                    model_file,
                    model_input,
                    model_output,
                )

            workers_used += len(gpus)

            zipfile_name = str(frame_batch) + ".zip"
            if upscale_dir:
                zipfile_name = os.path.join(upscale_dir, zipfile_name)

            logging.info("Zipping png files into " + zipfile_name)

            try:
                with zipfile.ZipFile(
                    zipfile_name,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                    compresslevel=0,
                ) as zip:
                    for frame in range(frame_range[0], frame_range[1] + 1):
                        zip.write(str(frame) + ".png")
            except Exception as e:
                logging.error("Zipfile creation failed")
                logging.error(e)
                sys.exit("Error - Exiting")

            for frame in range(frame_range[0], frame_range[1] + 1):
                os.remove(str(frame) + ".png")

        with open("upscaled.txt", "w") as f:
            f.write("Upscaled")

        logging.info("Upscale only finished for " + input_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Upscale images only")

    parser.add_argument("-i", "--input_file", required=True, help="Input file.")
    parser.add_argument("-f", "--ffmpeg", required=True, help="Location of ffmpeg.")
    parser.add_argument(
        "-m",
        "--models",
        help="Adds additional processing. 'a' for anime videos, 'n={denoise level}' for noise reduction and 'r' for real life imaging. Example: -m a,n=3,r to use all three options.",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        default=2,
        help="Scale 1, 2 or 4. Default is 2. If using real life imaging (4x model), scale will autoset to 4.",
    )
    parser.add_argument(
        "-t", "--temp_dir", help="Temp directory. Default is tempfile.gettempdir()."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=10,
        help="Number of minutes to upscale per batch. Default is 10.",
    )
    parser.add_argument(
        "-g", "--gpus", help="Optional gpus to use. Example 0,1,1,2. Default is 0."
    )
    parser.add_argument(
        "-u", "--upscale_dir", help="Upscale directory. Default is same as temp_dir."
    )
    parser.add_argument(
        "-x",
        "--extract_only",
        action="store_true",
        help="Exits after frame extraction. Used in conjunction with --resume_processing. You may want to run test_image.py on some extracted png files to sample what denoise level to apply if needed. Rerun with -r / --resume_processing to restart.",
    )
    parser.add_argument(
        "-l", "--log_level", type=int, help="Logging level. logging.INFO is default"
    )
    parser.add_argument("-d", "--log_dir", help="Logging directory. logging directory")

    args = parser.parse_args()

    upscale_only(
        args.input_file,
        args.ffmpeg,
        args.scale,
        args.temp_dir,
        args.batch_size,
        args.gpus,
        args.upscale_dir,
        args.extract_only,
        args.models,
        args.log_level,
        args.log_dir,
    )
