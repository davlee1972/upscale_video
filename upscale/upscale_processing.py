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
import cv2
from ncnn_vulkan import ncnn
import numpy as np
import math
import json
from multiprocessing import Pool


def get_metadata(ffmpeg, input_file):
    logging.info("Getting metadata from " + str(input_file))
    if os.path.exists("metadata.json"):
        info_dict = json.loads(open("metadata.json").read())
        frames_count = info_dict["number_of_frames"]
        duration = info_dict["duration"]
        frame_rate = info_dict["frame_rate"]
    else:
        result = subprocess.run(
            [
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
            ],
            capture_output=True,
            text=True,
        )

        if result.stderr:
            logging.error("Error getting metadata.")
            logging.error(str(result.stderr))
            logging.error(str(result.args))
            sys.exit("Error getting metadata.")

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

    frame_rate_check = round(duration * frame_rate / frames_count, 2)
    if frame_rate_check != 1:
        logging.info(
            "Frame rates mismatch detected: "
            + str(round(frames_count / duration, 2))
            + " vs "
            + str(round(frame_rate, 2))
        )
        logging.info(
            "Will attempt to adjust frame rate and number of frames to extract"
        )
        info_dict["number_of_frames"] = round(frames_count * frame_rate_check, 0)
        for i in range(1, 10):
            test = frame_rate_check * i
            if round(test, 0) == round(test, 2) and round(test - i, 2) == 1:
                test = int(test)
                info_dict["prune"] = (
                    "fps="
                    + str(info_dict["frame_rate"])
                    + ",fieldmatch=order=tff:mode=pc,decimate=cycle="
                    + str(test)
                )
                ##info_dict["prune"] = "fps=" + str(info_dict["frame_rate"])
                info_dict["frame_rate"] = round(frames_count / duration, 4)
                info_dict["number_of_frames"] = int(
                    info_dict["streams"][0]["nb_read_packets"]
                )
                logging.info(
                    "Corrected framerate is: " + str(round(info_dict["frame_rate"]))
                )
                logging.info(
                    "1 out of every " + str(test) + " duplicate frames will be pruned.."
                )
                break

    return info_dict


def get_crop_detect(ffmpeg, input_file, temp_dir):
    logging.info("Getting crop_detect from " + str(input_file))

    if os.path.exists("crop_detect.txt"):
        with open("crop_detect.txt") as f:
            crop = f.read()
    else:

        path, file_name = os.path.split(input_file)
        os.chdir(path)

        result = subprocess.run(
            [
                ffmpeg[:-6] + "ffprobe",
                "-hide_banner",
                "-v",
                "quiet",
                "-f",
                "lavfi",
                "-i",
                "movie=" + file_name + ",cropdetect",
                "-show_entries",
                "packet_tags=lavfi.cropdetect.w,lavfi.cropdetect.h,lavfi.cropdetect.x,lavfi.cropdetect.y",
                "-print_format",
                "json",
                "-loglevel",
                "error",
            ],
            capture_output=True,
            text=True,
        )

        os.chdir(temp_dir)

        if result.stderr:
            logging.error("Error with getting crop detect.")
            logging.error(str(result.stderr))
            logging.error(str(result.args))
            sys.exit("Error with getting crop detect.")

        crop_list = json.loads(result.stdout)["packets"]
        crop_list = [
            "crop=" + ":".join(row["tags"].values())
            for row in crop_list
            if "tags" in row and row["tags"] != {}
        ]
        crop = max(set(crop_list), key=crop_list.count)

        with open("crop_detect.txt", "w") as f:
            f.write(crop)

    return crop


def process_denoise(input_file_name, output_file_name, denoise, remove=True):

    img = cv2.UMat(cv2.imread(input_file_name))

    output = cv2.fastNlMeansDenoisingColored(img, None, denoise, 10, 5, 9)

    cv2.imwrite(output_file_name, output)

    if remove:
        os.remove(input_file_name)

    return "Processed Denoise: " + output_file_name


def process_model(input_file, output_file, net, input_name, output_name, remove=True):

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
        ex.input(input_name, mat_in)
        ret, mat_out = ex.extract(output_name)
        out = np.array(mat_out)

        # Transpose the output from `c, h, w` to `h, w, c` and put it back in 0-255 range
        output = out.transpose(1, 2, 0) * 255

        # Save image using opencv
        cv2.imwrite(output_file, output)
    except RuntimeError as error:
        loggin.error("Model processing failed")
        logging.error(error)
        ncnn.destroy_gpu_instance()
        sys.exit("Model processing failed")

    if remove:
        os.remove(input_file)

    logging.info("Processed Model: " + output_file)


def process_tile(
    net, img, input_name, output_name, tile_size, scale, y, x, height, width, output
):
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
        ex.input(input_name, mat_in)
        ret, mat_out = ex.extract(output_name)
        output_tile = np.array(mat_out)
    except RuntimeError as error:
        logging.error("Upscale failed")
        logging.error(error)
        ncnn.destroy_gpu_instance()
        sys.exit("Upscale failed")

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
    input_file_name, output_file_name, scale, net, input_name, output_name
):

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
            logging.debug(f"\tProcessing Tile: {tile_idx}/{tiles_x * tiles_y}")
            process_tile(
                net,
                img,
                input_name,
                output_name,
                tile_size,
                scale,
                y,
                x,
                height,
                width,
                output,
            )
    if output_file_name:
        cv2.imwrite(output_file_name, output)


def merge_frames(
    ffmpeg,
    ffmpeg_encoder,
    frame_batch,
    start_frame,
    end_frame,
    frame_rate,
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
        "p010le",
        "-global_quality",
        "20",
        "-loglevel",
        "error",
        str(frame_batch) + ".mkv",
    ]

    logging.info(
        "Merging Batch: "
        + str(frame_batch)
        + " : Number of frames: "
        + str(1 + end_frame - start_frame)
    )

    ## run ffmpeg to merge frames
    result = subprocess.run(cmds, capture_output=True, text=True)

    if result.stderr:
        logging.error("PNG merging Failed")
        logging.error(str(result.stderr))
        logging.error(str(result.args))

        return -1

    logging.info("Batch merged into " + str(frame_batch) + ".mkv")
    logging.info(str(end_frame) + " total frames upscaled")

    ## delete converted png files
    for i in range(start_frame, end_frame + 1):
        os.remove(str(i) + ".png")

    return 0


def upscale_frames(
    net,
    input_model_name,
    frame_batch,
    start_frame,
    end_frame,
    scale,
    input_name,
    output_name,
):

    logging.info(
        "Upscaling Batch: "
        + str(frame_batch)
        + " : Number of frames: "
        + str(1 + end_frame - start_frame)
    )

    frames_upscaled = 0

    ## upscale frames
    for frame in range(start_frame, end_frame + 1):

        output_file_name = str(frame) + ".png"
        input_file_name = str(frame) + "." + input_model_name + ".png"

        if os.path.exists(output_file_name):
            frames_upscaled += 1
            continue

        upscale_image(
            input_file_name, output_file_name, scale, net, input_name, output_name
        )

        os.remove(input_file_name)

        frames_upscaled += 1

        logging.info(
            "Upscaling Batch: "
            + str(frame_batch)
            + " : Upscaled "
            + str(frames_upscaled)
            + "/"
            + str(1 + end_frame - start_frame)
        )


def merge_mkvs(ffmpeg, frame_batches, output_file, log_dir):
    logging.info("Merging Fragments into " + output_file)
    with open("merge_list.txt", "w") as f:
        for i in range(frame_batches):
            f.write("file " + str(i + 1) + ".mkv\n")

    result = subprocess.run(
        [
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
        ],
        capture_output=True,
        text=True,
    )

    if result.stderr:
        logging.error("MKV merging failed")
        logging.error(str(result.stderr))
        logging.error(str(result.args))
        sys.exit("MKV merging failed")


def process_file(
    input_file,
    output_file,
    ffmpeg,
    ffmpeg_encoder,
    scale,
    temp_dir,
    batch_size,
    resume_processing,
    extract_only,
    anime,
    denoise,
    log_level,
    log_dir,
):
    """
    Upscale video file 2x or 4x

    :param input_file:
    :param output_file:
    :param ffmpeg:
    :param ffmpeg_encoder:
    :param scale:
    :param temp_dir:
    :param batch_size:
    :param resume_processing:
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

    if not output_file:
        output_file = input_file.split(".")
        output_file_ext = output_file[-1]
        output_file = ".".join(output_file[:-1] + [str(scale) + "x", output_file_ext])

    if log_dir:
        log_file = os.path.join(log_dir, output_file.split(os.sep)[-1][:-4] + ".log")
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
    if os.path.exists(temp_dir):
        if not resume_processing:
            shutil.rmtree(temp_dir)
            os.mkdir(temp_dir)
    else:
        os.mkdir(temp_dir)

    ## change working directory to temp directory
    cwd_dir = os.getcwd()
    os.chdir(temp_dir)

    if resume_processing and os.path.exists("completed.txt"):
        sys.exit(input_file + "already processed - Exiting")

    if sys.platform in ["win32", "cygwin", "darwin"]:
        from wakepy import set_keepawake

        set_keepawake(keep_screen_awake=False)

    ## get metadata
    info_dict = get_metadata(ffmpeg, input_file)

    frames_count = info_dict["number_of_frames"]
    frame_rate = info_dict["frame_rate"]

    ## calculate frames per minute
    frames_per_batch = int(frame_rate * 60) * batch_size

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
                        input_file_name,
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

        if (
            merge_frames(
                ffmpeg,
                ffmpeg_encoder,
                frame_batch,
                start_frame,
                end_frame,
                frame_rate,
            )
            == -1
        ):
            ncnn.destroy_gpu_instance()
            sys.exit("PNG merging Failed")

        frame_batch += 1

    ncnn.destroy_gpu_instance()
    del net

    ## merge video files into a single video file
    frame_batch -= 1
    merge_mkvs(ffmpeg, frame_batch, output_file, log_dir)

    with open("completed.txt", "w") as f:
        f.write("Completed")

    os.chdir(cwd_dir)

    logging.info("Upscale video finished for " + output_file)

    if not resume_processing:
        logging.info("Cleaning up temp directory")
        shutil.rmtree(temp_dir)
