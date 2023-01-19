"""
Copyright (c) 2022, David Lee
Author: David Lee
"""

import argparse
import logging
import os
import tempfile
import sys


from upscale_processing import (
    get_metadata,
    get_crop_detect,
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
    gpus,
    upscale_dir,
    extract_only,
    anime,
    denoise,
    log_level,
    log_dir,
):
    """
    Upscales video file 2x or 4x only

    :param input_file:
    :param ffmpeg:
    :param scale:
    :param temp_dir:
    :param gpus:
    :param extract_only:
    :param anime:
    :param denoise:
    :param log_level:
    :param log_dir:
    """

    if scale not in [2, 4]:
        sys.exit("Scale must be 2 or 4 - Exiting")

    if not os.path.exists(input_file):
        sys.exit(input_file + " not found")

    if upscale_dir and not os.path.exists(upscale_dir):
        sys.exit(upscale_dir + " is not not valid")

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

    if denoise:
        if denoise > 30:
            denoise = 30
        if denoise <= 0:
            denoise = None

    ## Create temp directory
    if not temp_dir:
        temp_dir = tempfile.gettempdir()

    temp_dir = os.path.abspath(os.path.join(temp_dir, "upscale_video"))
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    ## change working directory to temp directory
    cwd_dir = os.getcwd()
    os.chdir(temp_dir)

    if os.path.exists("upscaled.txt"):
        sys.exit(input_file + "already processed - Exiting")

    if sys.platform in ["win32", "cygwin", "darwin"]:
        from wakepy import set_keepawake

        set_keepawake(keep_screen_awake=False)

    ## get metadata
    info_dict = get_metadata(ffmpeg, input_file)

    frames_count = info_dict["number_of_frames"]
    frame_rate = info_dict["frame_rate"]
    duration = info_dict["duration"]

    ## calculate frames per minute
    frames_per_batch = int(frame_rate * 60)

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

    if anime:
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

    if denoise:
        logging.info("Starting denoise touchup...")

        workers_used += process_denoise(frames_count, input_file_tag, denoise)

        input_file_tag = "denoise"

    logging.info("Starting upscale processing...")

    ## process input file in batches
    upscale_frames(
        None,
        2,
        frames_count,
        input_file_tag,
        scale,
        gpus,
        workers_used,
        model_path,
        upscale_dir,
    )

    with open("upscaled.txt", "w") as f:
        f.write("Upscaled")

    os.chdir(cwd_dir)

    logging.info("Upscale only finished for " + input_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Upscale images only")

    parser.add_argument("-i", "--input_file", required=True, help="Input file.")
    parser.add_argument("-f", "--ffmpeg", required=True, help="Location of ffmpeg.")
    parser.add_argument(
        "-a",
        "--anime",
        action="store_true",
        help="Adds additional processing for anime videos to remove grain and smooth color.",
    )
    parser.add_argument(
        "-n",
        "--denoise",
        type=int,
        help="Adds additional processing to remove film grain. Denoise level 1 to 30. 3 = light / 10 = heavy.",
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=2, help="Scale 2 or 4. Default is 2."
    )
    parser.add_argument(
        "-t", "--temp_dir", help="Temp directory. Default is tempfile.gettempdir()."
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
        args.gpus,
        args.upscale_dir,
        args.extract_only,
        args.anime,
        args.denoise,
        args.log_level,
        args.log_dir,
    )
