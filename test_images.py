"""
Copyright (c) 2022, David Lee
Author: David Lee
"""
import logging
import os
import sys
import argparse
import shutil
from upscale.upscale_processing import (
    process_model,
    process_denoise,
    upscale_frames,
    get_frames,
)


def process_image(
    input_frames,
    temp_dir,
    output_dir,
    scale,
    anime,
    denoise,
    gpus,
):

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    if denoise:
        if denoise > 30:
            denoise = 30

        if denoise <= 0:
            denoise = None

    if scale not in [2, 4]:
        sys.exit("Scale must be 2 or 4")

    if gpus:
        try:
            gpus = gpus.split(",")
            gpus = [int(g) for g in gpus]
        except ValueError:
            logging.error("Invalid gpus")
            sys.exit("Error - Exiting")
    else:
        gpus = [0]

    model_path = os.path.realpath(__file__).split(os.sep)
    model_path = os.sep.join(model_path[:-1] + ["models"])

    input_frames = get_frames(input_frames)

    temp_dir = os.path.abspath(os.path.join(temp_dir, "upscale_video"))

    for frame in input_frames:
        shutil.copyfile(
            os.path.join(temp_dir, str(frame) + ".extract.png"),
            os.path.join(output_dir, str(frame) + ".extract.png"),
        )

    os.chdir(output_dir)

    workers_used = 0
    input_file_tag = "extract"

    if anime:
        logging.info("Starting anime touchup...")

        model_file = "x_HurrDeblur_SubCompact_nf24-nc8_244k_net_g"
        output_file_tag = "anime"

        process_model(
            input_frames,
            model_path,
            model_file,
            1,
            "input",
            "output",
            input_file_tag,
            output_file_tag,
            gpus,
            workers_used,
            remove=False,
        )

        workers_used += len(gpus)
        input_file_tag = "anime"

    if denoise:
        logging.info("Starting denoise touchup...")

        workers_used += process_denoise(input_frames, input_file_tag, denoise, remove=False)

        input_file_tag = "denoise"

    logging.info("Starting upscale processing...")

    for frame in input_frames:
        try:
            os.remove(str(frame) + ".png")
        except:
            pass

    upscale_frames(
        input_frames,
        frame,
        frame,
        input_file_tag,
        scale,
        gpus,
        workers_used,
        model_path,
        remove=False,
    )

    logging.info("Completed")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test Image Upscaler")
    parser.add_argument(
        "-i",
        "--input_frames",
        required=True,
        help="List of input frames in format like 1,3,5-7,10-12,15",
    )
    parser.add_argument(
        "-t",
        "--temp_dir",
        help="Temp directory where extracted frames are saved. Default is tempfile.gettempdir().",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Output directory where test images will be saved",
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=2, help="scale 2 or 4. Default is 2."
    )
    parser.add_argument(
        "-a",
        "--anime",
        action="store_true",
        help="Adds processing for anime to remove grain and smooth color.",
    )
    parser.add_argument(
        "-n",
        "--denoise",
        type=int,
        default=0,
        help="Adds processing to reduce image grain. Denoise level 1 to 30. 3 = light / 10 = heavy, etc..",
    )
    parser.add_argument(
        "-g", "--gpus", help="Optional gpu #s to use. Example 0,1,3. Default is 0."
    )

    args = parser.parse_args()

    process_image(
        args.input_frames,
        args.temp_dir,
        args.output_dir,
        args.scale,
        args.anime,
        args.denoise,
        args.gpus,
    )
