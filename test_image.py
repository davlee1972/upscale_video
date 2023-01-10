"""
Copyright (c) 2022, David Lee
Author: David Lee
"""
import logging
import os
import sys
from ncnn_vulkan import ncnn
import argparse
from upscale.upscale_processing import process_model, process_denoise, upscale_image


def process_image(
    input_file,
    output_file,
    scale,
    anime,
    denoise,
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

    net = ncnn.Net()

    # Use vulkan compute
    net.opt.use_vulkan_compute = True

    model_path = os.path.realpath(__file__).split(os.sep)
    model_path = os.sep.join(model_path[:-1] + ["models"])

    logging.info("Processing File: " + input_file)

    if not output_file:
        output_file = input_file.split(".")
        output_file_ext = output_file[-1]
        output_file = ".".join(output_file[:-1] + [str(scale) + "x", output_file_ext])

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

        process_model(
            input_file,
            input_file[:-4] + ".anime.png",
            net,
            input_name,
            output_name,
        )
        input_file = input_file[:-4] + ".anime.png"

    if denoise:
        logging.info("Starting denoise touchup...")

        process_denoise(
            input_file,
            input_file[:-4] + ".denoise.png",
            denoise,
            remove=False,
        )
        input_file = input_file[:-4] + ".denoise.png"

    net.load_param(os.path.join(model_path, str(scale) + "x_Compact_Pretrain.param"))
    net.load_model(os.path.join(model_path, str(scale) + "x_Compact_Pretrain.bin"))
    input_name = "input"
    output_name = "output"

    for frame in bad_frames:
        input_file_name = str(frame) + "." + input_model_name + ".png"
        output_file_name = str(frame) + ".png"
        upscale_image(
            input_file, output_file, scale, net, input_name, output_name, remove=False
        )

    logging.info("Completed")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test Image Upscaler")
    parser.add_argument("-i", "--input_file", required=True, help="Input image file.")
    parser.add_argument(
        "-o",
        "--output_file",
        help="Optional output image file. Default is input_file + ('.2x.' or '.4x.')",
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
        "-s", "--scale", type=int, default=2, help="scale 2 or 4. Default is 2."
    )
    args = parser.parse_args()

    process_image(
        args.input_file,
        args.output_file,
        args.scale,
        args.anime,
        args.denoise,
    )
