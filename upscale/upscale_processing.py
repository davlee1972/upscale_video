"""
Copyright (c) 2022, David Lee
Author: David Lee
"""
import logging
import os
import subprocess
import tempfile
import shutil
import sys
import math
import json
import multiprocessing
import time

import cv2
from ncnn_vulkan import ncnn
import numpy as np
from PIL import Image
from wakepy import keep

net = None
model_input_name = "input"
model_output_name = "output"


def get_frames(x):
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


def logging_callback(log_list):
    error_check = False
    for level, message in log_list:
        if level == "info":
            logging.info(message)
        elif level == "debug":
            logging.debug(message)
        elif level == "error":
            logging.error(message)
            error_check = True
        if error_check:
            sys.exit("Error - Exiting")


def init_worker(
    gpus, workers_used, model_path, model_file, scale, model_input, model_output
):
    global net, model_input_name, model_output_name

    gpu = multiprocessing.current_process()._identity[0] - 1 - workers_used

    if gpu > len(gpus) - 1:
        logging.error("Unable to assign GPU to new worker.")
        sys.exit("Error - Exiting")

    net = ncnn.Net()

    net.opt.use_vulkan_compute = True
    net.set_vulkan_device(gpus[gpu])

    net.load_param(os.path.join(model_path, str(scale) + model_file + ".param"))
    net.load_model(os.path.join(model_path, str(scale) + model_file) + ".bin")
    model_input_name = model_input
    model_output_name = model_output


def get_metadata(ffmpeg, input_file):
    if input_file:
        logging.info("Getting metadata from " + str(input_file))
    else:
        logging.info("Getting metadata")

    if os.path.exists("metadata.json"):
        info_dict = json.loads(open("metadata.json").read())
        frames_count = info_dict["number_of_frames"]
        duration = info_dict["duration"]
        frame_rate = info_dict["frame_rate"]
    else:
        cmds = [
            ffmpeg[:-6] + "ffprobe",
            "-hide_banner",
            "-v",
            "quiet",
            "-show_format",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets,r_frame_rate",
            "-print_format",
            "json",
            "-loglevel",
            "error",
            "-i",
            input_file,
        ]

        logging.info(cmds)

        result = subprocess.run(cmds, capture_output=True, text=True)

        if result.stderr:
            logging.error("Error getting metadata.")
            logging.error(str(result.stderr))
            logging.error(str(result.args))
            sys.exit("Error - Exiting")

        info_dict = json.loads(result.stdout)

        duration = float(info_dict["format"]["duration"])
        frames_count = int(info_dict["streams"][0]["nb_read_packets"])
        frame_rate = eval(info_dict["streams"][0]["r_frame_rate"])

        info_dict["number_of_frames"] = frames_count
        info_dict["duration"] = duration
        info_dict["frame_rate"] = frame_rate

        with open("metadata.json", "w") as f:
            f.write(json.dumps(info_dict))

    logging.info("Number of frames: " + str(frames_count))
    logging.info("Duration: " + str(duration))
    logging.info("Frames per second: " + str(frame_rate))

    return info_dict


def get_crop_detect(ffmpeg, input_file, duration):
    logging.info("Getting crop_detect from " + str(input_file))

    if os.path.exists("crop_detect.txt"):
        with open("crop_detect.txt") as f:
            crop = f.read()
    else:
        interval = int(duration / 120)
        crop_list = []

        for i in range(10,110):
            cmds = [
                ffmpeg,
                "-hide_banner",
                "-ss",
                str((i + 1) * interval),
                "-i",
                input_file,
                "-frames:v",
                "2",
                "-vf",
                "cropdetect",
                "-f",
                "null",
                "-",
            ]
            logging.info(cmds)
            result = subprocess.run(cmds, capture_output=True, text=True)
            lines = result.stderr.split("\n")
            for line in lines:
                if "crop=" in line:
                    crop = [
                        crop for crop in line.split(" ") if crop.startswith("crop=")
                    ][0].rstrip()
                    crop_list.append(crop)

        if crop_list:
            crop = max(set(crop_list), key=crop_list.count)
        else:
            crop = ""

        with open("crop_detect.txt", "w") as f:
            f.write(crop)

    return crop


def calc_batches(frames_count, batch_size):
    end_frame = 0
    frame_batch = 1
    frame_batches = {}
    while end_frame < frames_count:
        if frame_batch * batch_size < frames_count:
            end_frame = frame_batch * batch_size
        else:
            end_frame = frames_count

        start_frame = 1 + (frame_batch - 1) * batch_size

        frame_batches[frame_batch] = [start_frame, end_frame]

        frame_batch += 1

    return frame_batches


def extract_frames(
    ffmpeg,
    input_file,
    crop_detect,
    info_dict,
    frames_count,
    frame_batches,
    extract_only,
    output_format="mkv",
):

    cmds = [
        ffmpeg,
        "-hide_banner",
        "-hwaccel",
        "auto",
        "-i",
        input_file,
        "-loglevel",
        "error",
        "-pix_fmt",
        "rgb24",
    ]

    if crop_detect:
        logging.info("Crop Detected: " + crop_detect)
        cmds.append("-vf")
        cmds.append(crop_detect)

    cmds.append("%d.extract.png")

    ## Extract frames to temp dir. Need 300 gigs for a 2 hour movie
    logging.info("Starting Frames Extraction..")

    if extract_only or (
        not os.path.exists(str(frames_count) + ".extract.png")
        and not os.path.exists(str(frames_count) + ".anime.png")
        and not os.path.exists(str(frames_count) + ".denoise.png")
        and not os.path.exists(str(max(frame_batches.keys())) + "." + output_format)
    ):

        logging.info(cmds)
        result = subprocess.run(cmds)

        if result.stderr:
            logging.error("Error with extracting frames.")
            logging.error(str(result.stderr))
            logging.error(str(result.args))
            sys.exit("Error - Exiting")

    if extract_only:
        logging.info("Extract Only - Frames Extraction Completed")
        sys.exit()


def apply_model(input_file, output_file, remove):

    logging_items = []

    # Load image using opencv
    img = cv2.imread(input_file)

    mat_in = ncnn.Mat.from_pixels(
        img,
        ncnn.Mat.PixelType.PIXEL_BGR,
        img.shape[1],
        img.shape[0],
    )
    mean_vals = []
    norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
    mat_in.substract_mean_normalize(mean_vals, norm_vals)

    # Try/except block to catch out-of-memory error
    try:
        # Make sure the input and output names match the param file
        ex = net.create_extractor()
        ex.input(model_input_name, mat_in)
        ret, mat_out = ex.extract(model_output_name)
        out = np.array(mat_out)

        # Transpose the output from `c, h, w` to `h, w, c` and put it back in 0-255 range
        output = out.transpose(1, 2, 0) * 255

        # Save image using opencv
        if output_file:
            cv2.imwrite(output_file, output)
    except Exception as e:
        logging_items.append(["error", "Model processing failed"])
        logging_items.append(["error", e])
        ncnn.destroy_gpu_instance()
        return logging_items

    if remove:
        os.remove(input_file)

    logging_items.append(["info", "Processed Model: " + output_file])
    return logging_items


def process_model(
    frames_count,
    model_path,
    model_file,
    scale,
    model_input,
    model_output,
    input_file_tag,
    output_file_tag,
    gpus,
    workers_used,
    remove=True,
):

    if isinstance(frames_count, int):
        frames = range(1, frames_count + 1)
    else:
        frames = frames_count

    pool = multiprocessing.get_context("spawn").Pool(
        processes=len(gpus),
        initializer=init_worker,
        initargs=(
            gpus,
            workers_used,
            model_path,
            model_file,
            scale,
            model_input,
            model_output,
        ),
    )

    for frame in frames:
        input_file_name = str(frame) + "." + input_file_tag + ".png"
        output_file_name = str(frame) + "." + output_file_tag + ".png"

        if os.path.exists(input_file_name):
            pool.apply_async(
                apply_model,
                args=(input_file_name, output_file_name, remove),
                callback=logging_callback,
            )

    pool.close()
    pool.join()


def apply_denoise(input_file_name, output_file_name, denoise, remove):

    img = cv2.UMat(cv2.imread(input_file_name))

    output = cv2.fastNlMeansDenoisingColored(img, None, denoise, denoise, 5, 9)

    cv2.imwrite(output_file_name, output)

    if remove:
        os.remove(input_file_name)

    return [["info", "Processed Denoise: " + output_file_name]]


def process_denoise(frames_count, input_file_tag, denoise, remove=True):

    if isinstance(frames_count, int):
        frames = range(1, frames_count + 1)
    else:
        frames = frames_count

    pool = multiprocessing.get_context("spawn").Pool()

    for frame in frames:
        input_file_name = str(frame) + "." + input_file_tag + ".png"
        output_file_name = str(frame) + ".denoise.png"

        if os.path.exists(input_file_name):
            pool.apply_async(
                apply_denoise,
                args=(
                    input_file_name,
                    output_file_name,
                    denoise,
                    remove,
                ),
                callback=logging_callback,
            )

    pool.close()
    pool.join()

    return pool._processes


def process_tile(img, tile_size, scale, y, x, height, width, output, logging_items):

    # extract tile from input image
    ofs_y = y * tile_size
    ofs_x = x * tile_size

    # input tile area on total image
    input_start_y = ofs_y
    input_end_y = min(ofs_y + tile_size, height)
    input_start_x = ofs_x
    input_end_x = min(ofs_x + tile_size, width)

    # calculate borders to help ai scale between tiles

    if input_start_y >= 10:
        b_start_y = -10
    else:
        b_start_y = 0

    if input_end_y <= height - 10:
        b_end_y = 10
    else:
        b_end_y = 0

    if input_start_x >= 10:
        b_start_x = -10
    else:
        b_start_x = 0

    if input_end_x <= width - 10:
        b_end_x = 10
    else:
        b_end_x = 0

    # input tile dimensions
    input_tile = img[
        input_start_y + b_start_y : input_end_y + b_end_y,
        input_start_x + b_start_x : input_end_x + b_end_x,
        :,
    ].copy()

    # Convert image to ncnn Mat
    mat_in = ncnn.Mat.from_pixels(
        input_tile,
        ncnn.Mat.PixelType.PIXEL_BGR,
        input_tile.shape[1],
        input_tile.shape[0],
    )
    mean_vals = []
    norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
    mat_in.substract_mean_normalize(mean_vals, norm_vals)

    # upscale tile
    try:
        # Make sure the input and output names match the param file
        ex = net.create_extractor()
        ex.input(model_input_name, mat_in)
        ret, mat_out = ex.extract(model_output_name)
        output_tile = np.array(mat_out)
    except Exception as e:
        logging_items.append(["error", "Upscale failed"])
        logging_items.append(["error", e])
        logging.error(e)
        ncnn.destroy_gpu_instance()
        return -1

    # Transpose the output from `c, h, w` to `h, w, c` and put it back in 0-255 range
    output_tile = output_tile.transpose(1, 2, 0) * 255

    # scale area on total image
    input_start_y = input_start_y * scale
    b_start_y = b_start_y * scale
    input_end_y = input_end_y * scale
    input_start_x = input_start_x * scale
    b_start_x = b_start_x * scale
    input_end_x = input_end_x * scale

    ## transpose time
    output[input_start_y:input_end_y, input_start_x:input_end_x, :] = output_tile[
        -1 * b_start_y : input_end_y - input_start_y - b_start_y,
        -1 * b_start_x : input_end_x - input_start_x - b_start_x,
        :,
    ]


def upscale_image(
    input_file_name, output_file_name, scale, frame_batch, frame, end_frame, remove=True
):

    logging_items = []

    # Load image using opencv
    img = cv2.imread(input_file_name)

    tile_size = 960

    height, width, batch = img.shape
    output_height = height * scale
    output_width = width * scale
    output_shape = (output_height, output_width, batch)

    # start with black image
    output = np.zeros(output_shape)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for y in range(tiles_y):
        for x in range(tiles_x):
            tile_idx = y * tiles_x + x + 1

            logging_items.append(
                ["debug", f"Processing Tile: {tile_idx}/{tiles_x * tiles_y}"]
            )

            if (
                process_tile(
                    img, tile_size, scale, y, x, height, width, output, logging_items
                )
                == -1
            ):
                return logging_items

    if output_file_name:
        cv2.imwrite(output_file_name, output)

    if remove:
        os.remove(input_file_name)

    if frame_batch:
        if isinstance(frame_batch, int):
            logging_items.append(
                [
                    "info",
                    "Upscaling Batch: "
                    + str(frame_batch)
                    + " : Upscaled "
                    + str(frame)
                    + "/"
                    + str(end_frame),
                ]
            )
        else:
            logging_items.append(["info", "Upscaled " + output_file_name])
    else:
        logging_items.append(["info", "Upscaled " + str(frame) + "/" + str(end_frame)])

    return logging_items


def upscale_frames(
    frame_batch,
    start_frame,
    end_frame,
    input_file_tag,
    scale,
    gpus,
    workers_used,
    model_path,
    model_file,
    model_input,
    model_output,
    remove=True,
):

    if frame_batch and isinstance(frame_batch, list):
        frames = frame_batch
    else:
        frames = range(start_frame, end_frame + 1)

    pool = multiprocessing.get_context("spawn").Pool(
        processes=len(gpus),
        initializer=init_worker,
        initargs=(
            gpus,
            workers_used,
            model_path,
            model_file,
            scale,
            model_input,
            model_output,
        ),
    )

    ## upscale frames
    for frame in frames:

        input_file_name = str(frame) + "." + input_file_tag + ".png"
        output_file_name = str(frame) + ".png"

        if os.path.exists(input_file_name):
            pool.apply_async(
                upscale_image,
                args=(
                    input_file_name,
                    output_file_name,
                    scale,
                    frame_batch,
                    frame,
                    end_frame,
                    remove,
                ),
                callback=logging_callback,
            )

    pool.close()
    pool.join()


def merge_frames(
    ffmpeg,
    ffmpeg_encoder,
    frame_batch,
    start_frame,
    end_frame,
    frame_rate,
    pix_fmt,
    output_format,
):

    cmds = [
        ffmpeg,
        "-hide_banner",
        "-hwaccel",
        "auto",
        "-r",
        str(frame_rate),
        "-f",
        "image2",
        "-start_number",
        str(start_frame),
        "-i",
        "%d.png",
        "-vcodec",
        ffmpeg_encoder,
        "-frames:v",
        str(1 + end_frame - start_frame),
        "-pix_fmt",
        pix_fmt,
        "-global_quality",
        "20",
        "-loglevel",
        "error",
        str(frame_batch) + "." + output_format,
    ]

    logging.info(
        "Merging Batch: "
        + str(frame_batch)
        + " : Number of frames: "
        + str(1 + end_frame - start_frame)
    )

    ## run ffmpeg to merge frames
    logging.info(cmds)
    result = subprocess.run(cmds, capture_output=True, text=True)

    if result.stderr:
        if os.path.exists(str(frame_batch) + "." + output_format):
            os.remove(str(frame_batch) + "." + output_format)
        logging.error("PNG merging failed")
        logging.error(str(result.stderr))
        logging.error(str(result.args))
        logging.error("Testing PNG files for corruption..")
        bad_frames = []
        for frame in range(start_frame, end_frame + 1):
            try:
                img = Image.open(str(frame) + ".png")
                img.verify()
            except (IOError, SyntaxError) as e:
                logging.error("Bad file: " + str(frame) + ".png")
                bad_frames.append(frame)
                pass
        logging.error(
            "PNG merging failed - Try running fix_frames.py on bad frames using -b "
            + str(bad_frames)[1:-1].replace(" ", "")
        )
        sys.exit("Error - Exiting")

    time.sleep(5)

    if os.path.exists(str(frame_batch) + "." + output_format):
        logging.info("Batch merged into " + str(frame_batch) + "." + output_format)
        logging.info(str(end_frame) + " total frames merged")

        ## delete merged png files
        for frame in range(start_frame, end_frame + 1):
            os.remove(str(frame) + ".png")
    else:
        logging.error("Something went wrong with PNG merging..")
        logging.error(str(frame_batch) + "." + output_format + " not found..")
        sys.exit("Error - Exiting")


def merge_files(ffmpeg, frame_batches, output_file, output_format, log_dir):
    logging.info("Merging Fragments into " + output_file)
    output_format = output_file.split(".")[-1]
    with open("merge_list.txt", "w") as f:
        for i in range(frame_batches):
            f.write("file " + str(i + 1) + "." + output_format + "\n")

    cmds = [
        ffmpeg,
        "-hide_banner",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "merge_list.txt",
        "-loglevel",
        "error",
        "-c",
        "copy",
        output_file,
    ]

    logging.info(cmds)
    result = subprocess.run(cmds, capture_output=True, text=True)

    if result.stderr:
        if os.path.exists(output_file):
            os.remove(output_file)
        logging.error("File merging failed")
        logging.error(str(result.stderr))
        logging.error(str(result.args))
        sys.exit("Error - Exiting")

    ## delete merged files
    if os.path.exists(output_file):
        for i in range(frame_batches):
            os.remove(str(i + 1) + "." + output_format)
    else:
        logging.error("Something went wrong with file merging..")
        logging.error(output_file + " not found..")
        sys.exit("Error - Exiting")


def process_file(
    input_file,
    output_file,
    ffmpeg,
    ffmpeg_encoder,
    pix_fmt,
    scale,
    temp_dir,
    batch_size,
    gpus,
    resume_processing,
    extract_only,
    models,
    log_level,
    log_dir,
):
    """
    Upscale video file 2x or 4x

    :param input_file:
    :param output_file:
    :param ffmpeg:
    :param ffmpeg_encoder:
    :param pix_fmt:
    :param scale:
    :param temp_dir:
    :param batch_size:
    :param gpus:
    :param resume_processing:
    :param extract_only:
    :param models:
    :param log_level:
    :param log_dir:
    """

    if scale not in [1, 2, 4]:
        sys.exit("Scale must be 1, 2 or 4 - Exiting")

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

    output_format = input_file.split(".")[-1]

    if not output_file:
        output_file = input_file.split(".")
        output_file = ".".join(output_file[:-1] + [str(scale) + "x", output_format])

    logging.info("Processing File: " + input_file)

    ## Create temp directory
    if not temp_dir:
        temp_dir = tempfile.gettempdir()

    start_dir = temp_dir

    temp_dir = os.path.abspath(os.path.join(temp_dir, "upscale_video"))
    if os.path.exists(temp_dir):
        if not resume_processing:
            shutil.rmtree(temp_dir)
            os.mkdir(temp_dir)
    else:
        os.mkdir(temp_dir)

    ## change working directory to temp directory
    os.chdir(temp_dir)

    if resume_processing and os.path.exists("completed.txt"):
        sys.exit(input_file + " already processed - Exiting")

    with keep.running() as m:

        ## get metadata
        info_dict = get_metadata(ffmpeg, input_file)

        frames_count = info_dict["number_of_frames"]
        frame_rate = info_dict["frame_rate"]
        duration = info_dict["duration"]

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
            frame_batches,
            extract_only,
            output_format,
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
                remove=True,
            )

            workers_used += len(gpus)
            input_file_tag = "anime"

        logging.info("Starting upscale processing...")

        if "r" in models:
            model_file = "x_Valar_v1"
            model_input = "input"
            model_output = "output"
        else:
            model_file = "x_Compact_Pretrain"
            model_input = "input"
            model_output = "output"

        ## process input file in batches:
        for frame_batch, frame_range in frame_batches.items():

            if os.path.exists(str(frame_batch) + "." + output_format):
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

            merge_frames(
                ffmpeg,
                ffmpeg_encoder,
                frame_batch,
                frame_range[0],
                frame_range[1],
                frame_rate,
                pix_fmt,
                output_format,
            )

        ## merge video files into a single video file
        merge_files(ffmpeg, frame_batch, output_file, output_format, log_dir)

        with open("completed.txt", "w") as f:
            f.write("Completed")

        logging.info("Upscale video finished for " + output_file)

        if not resume_processing:
            logging.info("Cleaning up temp directory")
            os.chdir(start_dir)
            shutil.rmtree(temp_dir)
