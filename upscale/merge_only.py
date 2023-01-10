"""
Copyright (c) 2022, David Lee
Author: David Lee
"""

import argparse
import logging
import os
import tempfile
import sys

from upscale_processing import get_metadata, merge_frames, merge_mkvs, calc_batches


def merge_only(
    output_file,
    ffmpeg,
    ffmpeg_encoder,
    temp_dir,
    batch_size,
    log_level,
    log_dir,
):
    """
    Merge PNG files into Video File only

    :param output_file:
    :param ffmpeg:
    :param ffmpeg_encoder:
    :param temp_dir:
    :param batch_size:
    :param log_level:
    :param log_dir:
    """

    if not log_level:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    if log_dir:
        log_file = os.path.join(log_dir, output_file.split(os.sep)[-1][:-4] + ".log")
        # create log file handler and set level to debug
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        fh.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(fh)

    logging.info("Processing File: " + output_file)

    ## Create temp directory
    if not temp_dir:
        temp_dir = tempfile.gettempdir()

    temp_dir = os.path.abspath(os.path.join(temp_dir, "upscale_video"))

    ## change working directory to temp directory
    cwd_dir = os.getcwd()
    os.chdir(temp_dir)

    if os.path.exists("merged.txt"):
        sys.exit(output_file + "already processed - Exiting")

    if sys.platform in ["win32", "cygwin", "darwin"]:
        from wakepy import set_keepawake

        set_keepawake(keep_screen_awake=False)

    ## get metadata
    info_dict = get_metadata(ffmpeg, None)

    frames_count = info_dict["number_of_frames"]
    frame_rate = info_dict["frame_rate"]

    ## calculate frames per minute and frame_batches
    frames_per_batch = int(frame_rate * 60) * batch_size
    frame_batches = calc_batches(frames_count, frames_per_batch)

    for frame_batch, frame_range in frame_batches.items():

        if os.path.exists(str(frame_batch) + ".mkv"):
            frame_batch += 1
            continue

        for frame in range(frame_range[0], frame_range[1] + 1):
            input_file_name = str(frame) + ".png"
            if not os.path.exists(input_file_name):
                logging.error(input_file_name + " not found - Exiting")
                sys.exit()

        merge_frames(
            ffmpeg,
            ffmpeg_encoder,
            frame_batch,
            frame_range[0],
            frame_range[1],
            frame_rate,
        )

    ## merge video files into a single video file
    merge_mkvs(ffmpeg, frame_batch, output_file, log_dir)

    with open("merged.txt", "w") as f:
        f.write("Merged")

    os.chdir(cwd_dir)

    logging.info("Merge only finished for " + output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge images only")

    parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        help="Output video file location",
    )
    parser.add_argument("-f", "--ffmpeg", required=True, help="Location of ffmpeg.")
    parser.add_argument(
        "-e",
        "--ffmpeg_encoder",
        default="av1_qsv",
        help="ffmpeg encoder for mkv file. Default is av1_qsv.",
    )
    parser.add_argument(
        "-t", "--temp_dir", help="Temp directory. Default is tempfile.gettempdir()."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Number of minutes to upscale per batch. Default is 1.",
    )
    parser.add_argument(
        "-l", "--log_level", type=int, help="Logging level. logging.INFO is default"
    )
    parser.add_argument("-d", "--log_dir", help="Logging directory. logging directory")

    args = parser.parse_args()

    merge_only(
        args.output_file,
        args.ffmpeg,
        args.ffmpeg_encoder,
        args.temp_dir,
        args.batch_size,
        args.log_level,
        args.log_dir,
    )
