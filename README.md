# Upscales video 2x or 4x using AI

Compact Pretrained Models from the wiki below are used.

https://upscale.wiki/wiki/Model_Database#Real-ESRGAN_Compact_Custom_Models

This script will convert a video file using one minute's worth of frames at a time to save disk space. The video fragments are then merged at the end of the process into a final video file.

# You must have ffmpeg installed and a vulkan compatible GPU

Without a GPU this process would take weeks to process a 2 hour video
Running test_gpu.py will list out Vulkan compatible GPUs and other info.

Make sure you are passing in a ffmpeg encoder that is compatible with your system.
The default is "av1_qsv" since I'm using an Intel Arc A750 GPU which supports AV1 hardware encoding.

List of ffmpeg encoders is here: https://ffmpeg.org/ffmpeg-codecs.html#Video-Encoders.

# Installation

Download and install ffmpeg: https://ffmpeg.org/download.html

pip install ncnn_vulkan

pip install wakepy

git clone https://github.com/davlee1972/upscale_video.git

# Usage

-r / --resume_processing can be used after you abort a run and want to pickup where you left off.

-x / --extract_only will stop processing after frames extraction. You may want to run test_image.py
on some extracted png files to sample what denoise level (-n / --denoise) to apply if needed
and then resume processing (-r / --resume_processing) to pickup where you left off.

```console
Usage: python upscale_video.py -i INPUT_FILE -f FFMPEG

options:
  -h / --help                            Show this help message and exit
  -i / --input_file INPUT_FILE           Input video file.
  -o / --output_file OUTPUT_FILE         Optional output video file location.
                                         Default is input_file + ('.2x.' or '.4x.')
  -f / --ffmpeg FFMPEG                   Location of ffmpeg.
  -e / --ffmpeg_encoder FFMPEG_ENCODER   ffmpeg encoder for mkv file. Default is av1_qsv.
  -a / --anime                           Adds processing for anime to remove grain and color bleeding.
  -n / --denoise DENOISE                 Adds processing to remove film grain. Denoise level 1 to 30.
                                         3 = light / 10 = heavy, etc.
  -s / --scale SCALE                     Scale 2 or 4. Default is 2.
  -t / --temp_dir TEMP_DIR               Temp directory. Default is tempfile.gettempdir().
  -b / --batch_size BATCH_SIZE           Number of minutes to upscale per batch. Default is 1.
  -r / --resume_processing               Does not purge any data in temp_dir when restarting.
  -x / --extract_only                    Exits after frames extraction. You may want to
                                         run test_image.py on some extracted png files
                                         to sample what denoise level to apply if needed.
                                         Rerun with -r / --resume_processing to restart.
  -l / --log_level LOG_LEVEL             Logging level. logging.INFO is default
  -d / --log_dir LOG_DIR                 Logging directory. logging directory

```

```console
Usage: python test_image.py -i infile

options:
  -h / --help                      Show this help message and exit
  -i / --input_file INPUT_FILE     Input image file.
  -o / --output_file OUTPUT_FILE   Optional output image file.
                                   Default is input_file + ('.2x.' or '.4x.')
  -a / --anime                     Adds processing for anime to remove grain and smooth color.
  -n / --denoise DENOISE           Adds processing to reduce image grain. Denoise level 1 to 30.
                                   3 = light / 10 = heavy, etc.
  -s / --scale SCALE               Scale 2 or 4. Default is 2.

```

```console
Usage: python test_gpu.py -g gpu

options:
  -h / --help      Show this help message and exit
  -g / --gpu GPU   Optional gpu number to test. -1 to test cpu.

```


# Samples

![alt text](https://i.imgur.com/nkbA0Ft.png)
Original 1920 x 800 extracted image from Underworld Blu-ray

![alt text](https://i.imgur.com/Z2djqQN.png)
Upscaled 2x using --scale 2. Took 40 hours to process 200,000+ frames.

![alt text](https://i.imgur.com/GOFMK47.png)
Upscaled 2x with light denoise using --scale 2 --denoise 3. Denoise added an additional 3 hours of processing.

# Notes

This python code is used to scale my existing 2k bluray collecion to 4k

Source framerates are assumed to be more or less consistent. 
A frames per second calculation is performed taking total number of frames / duration.
It should come out to 23.976 fps for most movies.
This fps is used to reassemble the upscaled png images into a video file at the very end.
