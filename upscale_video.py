"""
Copyright (c) 2022, David Lee
Author: David Lee
"""
import argparse

from upscale.upscale_processing import process_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Upscale Video 2x or 4x")

    parser.add_argument("-i", "--input_file", required=True, help="Input video file.")
    parser.add_argument(
        "-o",
        "--output_file",
        help="Optional output video file location. Default is input_file + ('.2x.' or '.4x.')",
    )
    parser.add_argument("-f", "--ffmpeg", required=True, help="Location of ffmpeg.")
    parser.add_argument(
        "-e",
        "--ffmpeg_encoder",
        default="av1_qsv",
        help="ffmpeg encoder for video file. Default is av1_qsv.",
    )
    parser.add_argument(
        "-p",
        "--pix_fmt",
        default="p010le",
        help="pixel format used when merging frames into a video file. Default is p010le which is 10 bit color.",
    )
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
        "-r",
        "--resume_processing",
        action="store_true",
        help="Does not purge any data in temp_dir when restarting",
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

    process_file(
        args.input_file,
        args.output_file,
        args.ffmpeg,
        args.ffmpeg_encoder,
        args.scale,
        args.temp_dir,
        args.batch_size,
        args.gpus,
        args.resume_processing,
        args.extract_only,
        args.models,
        args.log_level,
        args.log_dir,
    )
