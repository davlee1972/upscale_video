"""
Copyright (c) 2022, David Lee
Author: David Lee
"""

import argparse
import logging
import os
import tempfile
import sys
import zipfile
import glob

from upscale_processing import get_metadata, merge_frames, merge_mkvs


def merge_only(
    output_file,
    ffmpeg,
    ffmpeg_encoder,
    temp_dir,
    log_level,
    log_dir,
):
    """
    Merge PNG files into Video File only

    :param output_file:
    :param ffmpeg:
    :param ffmpeg_encoder:
    :param temp_dir:
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

    frame_batch = 1

    while True:

        if os.path.exists(str(frame_batch) + ".mkv"):
            frame_batch += 1
            continue

        if os.path.exists(str(frame_batch) + ".zip"):
            logging.info("Extracting png files from " + str(frame_batch) + ".zip")
            with zipfile.ZipFile(str(frame_batch) + ".zip", "r") as zObject:
                try:
                    zObject.extractall()
                except Exception as e:
                    logging.error("Zipfile extract failed")
                    logging.error(e)
                    sys.exit("Error - Exiting")

            os.remove(str(frame_batch) + ".zip")

        png_files = glob.glob("*.png")
        png_files = [int(file_name.split(".")[0]) for file_name in png_files]

        if png_files:
            starting_frame = min(png_files)
            last_frame = max(png_files)
        else:
            logging.error("No more png files found - Exiting")
            sys.exit()

        if last_frame - starting_frame + 1 != len(png_files):
            logging.error("Frame counts mismatch - Exiting")
            logging.error(
                str(last_frame - starting_frame + 1)
                + " vs "
                + str(len(png_files))
                + " found."
            )
            sys.exit()

        merge_frames(
            ffmpeg,
            ffmpeg_encoder,
            frame_batch,
            starting_frame,
            last_frame,
            frame_rate,
        )

        if last_frame == frames_count:
            break

        frame_batch += 1

    ## merge video files into a single video file
    merge_mkvs(ffmpeg, frame_batch, output_file, log_dir)

    with open("merged.txt", "w") as f:
        f.write("Merged")

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
        "-l", "--log_level", type=int, help="Logging level. logging.INFO is default"
    )
    parser.add_argument("-d", "--log_dir", help="Logging directory. logging directory")

    args = parser.parse_args()

    merge_only(
        args.output_file,
        args.ffmpeg,
        args.ffmpeg_encoder,
        args.temp_dir,
        args.log_level,
        args.log_dir,
    )
