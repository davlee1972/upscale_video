"""
Copyright (c) 2022, David Lee
Author: David Lee
"""

import argparse
import logging
import os
import subprocess
import tempfile
import sys
from ncnn_vulkan import ncnn
from multiprocessing import Pool

from upscale_processing import (
    get_metadata,
    get_crop_detect,
    process_model,
    process_denoise,
    upscale_image,
)


def get_bad_frames(x):
    result = []
    for part in x.split(","):
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            result.extend(range(a, b + 1))
        else:
            a = int(part)
            result.append(a)
    return result


def fix_frames(
    input_file,
    bad_frames,
    ffmpeg,
    scale,
    temp_dir,
    anime,
    denoise,
    log_level,
    log_dir,
):
    """
    Upscale video file 2x or 4x

    :param input_file:
    :param bad_frames:
    :param ffmpeg:
    :param scale:
    :param temp_dir:
    :param anime:
    :param denoise:
    :param log_level:
    :param log_dir:
    """

    if scale not in [2, 4]:
        sys.exit("Scale must be 2 or 4 - Exiting")

    if not os.path.exists(input_file):
        sys.exit(input_file + " not found")

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

    if sys.platform in ["win32", "cygwin", "darwin"]:
        from wakepy import set_keepawake

        set_keepawake(keep_screen_awake=False)

    ## get metadata
    info_dict = get_metadata(ffmpeg, input_file)

    frame_rate = info_dict["frame_rate"]

    crop_detect = get_crop_detect(ffmpeg, input_file, temp_dir)

    bad_frames = get_bad_frames(bad_frames)

    max_frame = 0

    for frame in bad_frames:
        if not os.path.exists(str(frame) + ".extract.png") and frame > max_frame:
            max_frame = frame

    if max_frame:
        cmds = [
            ffmpeg,
            "-hide_banner",
            "-hwaccel",
            "auto",
            "-i",
            input_file,
            "-vframes",
            str(max_frame),
            "-loglevel",
            "error",
            "-pix_fmt",
            "rgb24",
        ]

        if crop_detect:
            logging.info("Crop Detected: " + crop_detect)
            cmds.append("-vf")
            if "prune" in info_dict:
                cmds.append(crop_detect + "," + info_dict["prune"])
            else:
                cmds.append(crop_detect)
        elif "prune" in info_dict:
            cmds.append("-vf")
            cmds.append(info_dict["prune"])

        cmds.append("%d.extract.png")

        ## Extract frames to temp dir. Need 300 gigs for a 2 hour movie
        logging.info("Starting Frames Extraction..")

        result = subprocess.run(cmds)

        if result.stderr:
            logging.error("Error with extracting frames.")
            logging.error(str(result.stderr))
            logging.error(str(result.args))
            sys.exit("Error - Exiting")

        ## Extract frames to temp dir. Need 300 gigs for a 2 hour movie
        logging.info("Removing extra extracted frames.")

        for frame in range(max_frame):
            if frame not in bad_frames:
                os.remove(str(frame) + ".extract.png")

    net = ncnn.Net()

    # Use vulkan compute if vulkan is supported
    gpu_info = ncnn.get_gpu_info()
    if gpu_info and gpu_info.type() in [0, 1, 2]:
        net.opt.use_vulkan_compute = True

    model_path = os.path.realpath(__file__).split(os.sep)
    model_path = os.sep.join(model_path[:-2] + ["models"])

    input_model_name = "extract"

    if anime:
        logging.info("Starting anime touchup...")
        net.load_param(
            os.path.join(
                model_path, "1x_HurrDeblur_SubCompact_nf24-nc8_244k_net_g.param"
            )
        )
        net.load_model(
            os.path.join(model_path, "1x_HurrDeblur_SubCompact_nf24-nc8_244k_net_g.bin")
        )
        input_name = "input"
        output_name = "output"

        for frame in bad_frames:
            input_file_name = str(frame) + "." + input_model_name + ".png"

            if os.path.exists(input_file_name):
                process_model(
                    input_file_name,
                    str(frame) + ".anime.png",
                    net,
                    input_name,
                    output_name,
                )

        input_model_name = "anime"

    if denoise:
        logging.info("Starting denoise touchup...")
        pool = Pool()

        for frame in bad_frames:
            input_file_name = str(frame) + "." + input_model_name + ".png"

            if os.path.exists(input_file_name):
                pool.apply_async(
                    process_denoise,
                    args=(
                        str(frame) + "." + input_model_name + ".png",
                        str(frame) + ".denoise.png",
                        denoise,
                    ),
                    callback=logging.info,
                )

        pool.close()
        pool.join()

        input_model_name = "denoise"

    logging.info("Starting upscale processing...")

    # Load model param and bin. Make sure input and output names match what is in the .param file
    net.load_param(os.path.join(model_path, str(scale) + "x_Compact_Pretrain.param"))
    net.load_model(os.path.join(model_path, str(scale) + "x_Compact_Pretrain.bin"))
    input_name = "input"
    output_name = "output"

    for frame in bad_frames:

        input_file_name = str(frame) + "." + input_model_name + ".png"
        output_file_name = str(frame) + ".png"

        upscale_image(
            input_file_name, output_file_name, scale, net, input_name, output_name
        )

        logging.info("Upscaled frame " + str(frame))

    os.chdir(cwd_dir)

    logging.info("Fix frames finished")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fix frames")

    parser.add_argument("-i", "--input_file", required=True, help="Input file.")
    parser.add_argument(
        "-b",
        "--bad_frames",
        required=True,
        help="List of bad frames in format like 1,3,5-7,10-12,15",
    )
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
        "-l", "--log_level", type=int, help="Logging level. logging.INFO is default"
    )
    parser.add_argument("-d", "--log_dir", help="Logging directory. logging directory")

    args = parser.parse_args()

    fix_frames(
        args.input_file,
        args.bad_frames,
        args.ffmpeg,
        args.scale,
        args.temp_dir,
        args.anime,
        args.denoise,
        args.log_level,
        args.log_dir,
    )
