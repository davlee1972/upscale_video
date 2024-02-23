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

from wakepy import keep

from upscale_processing import (
    get_metadata,
    get_crop_detect,
    process_model,
    process_denoise,
    upscale_frames,
    get_frames,
)


def fix_frames(
    input_file,
    bad_frames,
    ffmpeg,
    scale,
    temp_dir,
    gpus,
    models,
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
    :param gpus:
    :param models:
    :param log_level:
    :param log_dir:
    """

    if scale not in [2, 4]:
        sys.exit("Scale must be 2 or 4 - Exiting")

    if not os.path.exists(input_file):
        sys.exit(input_file + " not found")

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
    cwd_dir = os.getcwd()
    os.chdir(temp_dir)

    with keep.running() as m:

        ## get metadata
        info_dict = get_metadata(ffmpeg, input_file)

        frame_rate = info_dict["frame_rate"]
        duration = info_dict["duration"]

        crop_detect = get_crop_detect(ffmpeg, input_file, duration)

        bad_frames = get_frames(bad_frames)

        missing_frames = []
        missing_test = 1

        for frame in bad_frames:
            if not os.path.exists(str(frame) + ".extract.png"):
                missing_frames.append(frame)

        if denoise:
            missing_test += 1
            for frame in bad_frames:
                if not os.path.exists(str(frame) + ".denoise.png"):
                    missing_frames.append(frame)

        if "a" in models:
            missing_test += 1
            for frame in bad_frames:
                if not os.path.exists(str(frame) + ".anime.png"):
                    missing_frames.append(frame)

        max_frame = {
            frame: missing_frames.count(frame)
            for frame in missing_frames
            if missing_frames.count(frame) == missing_test
        }.keys()
        if max_frame:
            max_frame = max(max_frame)

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

            logging.info(cmds)
            result = subprocess.run(cmds)

            if result.stderr:
                logging.error("Error with extracting frames.")
                logging.error(str(result.stderr))
                logging.error(str(result.args))
                sys.exit("Error - Exiting")

            ## Extract frames to temp dir. Need 300 gigs for a 2 hour movie
            logging.info("Removing extra extracted frames.")

            for frame in range(max_frame):
                if frame + 1 not in bad_frames:
                    try:
                        os.remove(str(frame + 1) + ".extract.png")
                    except:
                        pass

        model_path = os.path.realpath(__file__).split(os.sep)
        model_path = os.sep.join(model_path[:-2] + ["models"])

        workers_used = 0
        input_file_tag = "extract"

        if denoise:
            logging.info("Starting denoise touchup...")
            workers_used += process_denoise(bad_frames, input_file_tag, denoise)
            input_file_tag = "denoise"

        if "a" in models:
            logging.info("Starting anime touchup...")

            model_file = "x_HurrDeblur_SubCompact_nf24-nc8_244k_net_g"
            output_file_tag = "anime"

            process_model(
                bad_frames,
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

        logging.info("Starting upscale processing...")

        for frame in bad_frames:
            try:
                os.remove(str(frame) + ".png")
            except:
                pass

        if "r" in models:
            model_file = "x_Valar_v1"
            model_input = "input"
            model_output = "output"
        else:
            model_file = "x_Compact_Pretrain"
            model_input = "input"
            model_output = "output"

        if scale == 1:
            for frame in bad_frames:
                os.rename(str(frame) + "." + input_file_tag + ".png", str(frame) + ".png")
        else:
            upscale_frames(
                bad_frames,
                None,
                None,
                input_file_tag,
                scale,
                gpus,
                workers_used,
                model_path,
                model_file,
                model_input,
                model_output,
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
        "-m",
        "--models",
        help="Adds additional processing. 'a' for anime videos, 'n={denoise level}' for noise reduction and 'r' for real life imaging. Example: -m a,n=3,r to use all three options.",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        default=2,
        help="Scale 2 or 4. Default is 2. If using real life imaging (4x model), scale will autoset to 4.",
    )
    parser.add_argument(
        "-t", "--temp_dir", help="Temp directory. Default is tempfile.gettempdir()."
    )
    parser.add_argument(
        "-g", "--gpus", help="Optional gpus to use. Example 0,1,1,2. Default is 0."
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
        args.gpus,
        args.models,
        args.log_level,
        args.log_dir,
    )
