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
    upscale_frames,
)


def upscale_only(
    input_file,
    ffmpeg,
    scale,
    temp_dir,
    extract_only,
    anime,
    denoise,
    log_level,
    log_dir,
):
    """
    Upscale video file 2x or 4x

    :param input_file:
    :param ffmpeg:
    :param scale:
    :param temp_dir:
    :param extract_only:
    :param anime:
    :param denoise:
    :param log_level:
    :param log_dir:
    """

    if scale not in [2, 4]:
        sys.exit("Scale must be 2 or 4 - Exiting")

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

    if os.path.exists("completed.txt"):
        sys.exit(input_file + "already processed - Exiting")

    if sys.platform in ["win32", "cygwin", "darwin"]:
        from wakepy import set_keepawake

        set_keepawake(keep_screen_awake=False)

    ## get metadata
    info_dict = get_metadata(ffmpeg, input_file)

    frames_count = info_dict["number_of_frames"]
    frame_rate = info_dict["frame_rate"]

    ## calculate frames per minute
    frames_per_batch = int(frame_rate * 60)

    crop_detect = get_crop_detect(ffmpeg, input_file, temp_dir)

    cmds = [
        ffmpeg,
        "-hide_banner",
        "-hwaccel",
        "auto",
        "-i",
        input_file,
        "-log_level",
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
        cmds.append(info_dict)

    cmds.append("%d.extract.png")

    ## Extract frames to temp dir. Need 300 gigs for a 2 hour movie
    logging.info("Starting Frames Extraction..")

    if (
        not os.path.exists(str(frames_count) + ".extract.png")
        and not os.path.exists(str(frames_count) + ".anime.png")
        and not os.path.exists(str(frames_count) + ".denoise.png")
    ):
        result = subprocess.run(cmds)

        if result.stderr:
            logging.error(str(result.stderr))
            logging.error(str(result.args))
            sys.exit("Error with extracting frames.")

    if extract_only:
        sys.exit("Extract Only - Frames Extraction Completed")

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

        for frame in range(frames_count):
            input_file_name = str(frame + 1) + "." + input_model_name + ".png"

            if os.path.exists(input_file_name):
                process_model(
                    input_file_name,
                    str(frame + 1) + ".anime.png",
                    net,
                    input_name,
                    output_name,
                )

        input_model_name = "anime"

    if denoise:
        logging.info("Starting denoise touchup...")
        pool = Pool()

        for frame in range(frames_count):
            input_file_name = str(frame + 1) + "." + input_model_name + ".png"

            if os.path.exists(input_file_name):
                pool.apply_async(
                    process_denoise,
                    args=(
                        str(frame + 1) + "." + input_model_name + ".png",
                        str(frame + 1) + ".denoise.png",
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

    frame_batch = 1
    end_frame = 0

    while end_frame < frames_count:
        if frame_batch * frames_per_batch < frames_count:
            end_frame = frame_batch * frames_per_batch
        else:
            end_frame = frames_count

        if os.path.exists(str(frame_batch) + ".mkv"):
            frame_batch += 1
            continue

        start_frame = 1 + (frame_batch - 1) * frames_per_batch

        upscale_frames(
            net,
            input_model_name,
            frame_batch,
            start_frame,
            end_frame,
            scale,
            input_name,
            output_name,
        )

        frame_batch += 1

    ncnn.destroy_gpu_instance()
    del net

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
        args.extract_only,
        args.anime,
        args.denoise,
        args.log_level,
        args.log_dir,
    )
