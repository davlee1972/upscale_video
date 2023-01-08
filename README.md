# Upscales video 2x or 4x using AI

Compact Pretrained Models from the wiki below are used.

https://upscale.wiki/wiki/Model_Database#Real-ESRGAN_Compact_Custom_Models

This script will convert a video file using one minute's worth of frames at a time to save disk space. The video fragments are then merged at the end of the process into a final video file.

# You must have ffmpeg installed and a vulkan compatible GPU

Without a GPU this process would take weeks to process a 2 hour video
Running test_gpu.py will list out Vulkan compatible GPUs and other info.

Make sure you are passing in a ffmpeg encoder that is compatible with your system.
The default is "av1_qsv" since I'm using an Intel Arc A750 GPU which supports AV1 hardware encoding.

Also make sure you have the latest Mesa drivers installed if on linux (see Notes below).

List of ffmpeg encoders is here: https://ffmpeg.org/ffmpeg-codecs.html#Video-Encoders.

# Installation

Python 3.6 through 3.10. 3.11 isn't supported yet as of Jan 8th, 2023.

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

Troubleshooting Ubuntu. It took me a while to figure out why my AMD gpu on my linux server wasn't getting
by Vulkan. Apparently you have to run your python code throught a terminal session in the GUI (pressing Alt-T).
Running the python script using a non-GUI shell (Ctrl-Alt-1 / SSH / Putty, etc.) doesn't work for some reason.

Installing the latest Open Mesa drivers probably helped as well.
https://launchpad.net/~oibaf/+archive/ubuntu/graphics-drivers


```console
sudo add-apt-repository ppa:oibaf/graphics-drivers
sudo apt update
sudo apt upgrade
```

**Output from running test_gpu.py**

```console
From a plain shell

python test_gpu.py
[2023-01-08 11:17:43] [INFO] Searching for Vulkan compatible GPUs
[2023-01-08 11:17:43] [INFO] ====================================
[2023-01-08 11:17:43] [INFO] GPU count: 1
[2023-01-08 11:17:43] [INFO] ====================================
[2023-01-08 11:17:43] [INFO] Default GPU: 0
[2023-01-08 11:17:43] [INFO] ====================================
[2023-01-08 11:17:43] [INFO] GPU 0: CPU / llvmpipe (LLVM 15.0.5, 256 bits)

```

```console
From terminal app within Ubuntu GUI

python test_gpu.py
[2023-01-08 11:20:36] [INFO] Searching for Vulkan compatible GPUs
[2023-01-08 11:20:36] [INFO] ====================================
[2023-01-08 11:20:36] [INFO] GPU count: 4
[2023-01-08 11:20:36] [INFO] ====================================
[2023-01-08 11:20:36] [INFO] Default GPU: 0
[2023-01-08 11:20:36] [INFO] ====================================
[2023-01-08 11:20:36] [INFO] GPU 0: Integrated / AMD Radeon Graphics (RADV RENOIR)
[2023-01-08 11:20:36] [INFO] GPU 1: CPU / llvmpipe (LLVM 15.0.5, 256 bits)
[2023-01-08 11:20:36] [INFO] GPU 2: CPU / llvmpipe (LLVM 15.0.5, 256 bits)
[2023-01-08 11:20:36] [INFO] GPU 3: Integrated / AMD Radeon Graphics (RADV RENOIR)

```
