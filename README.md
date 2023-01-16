# Upscales video 2x or 4x using AI

Compact Pretrained Models from the wiki below are used.

https://upscale.wiki/wiki/Model_Database#Real-ESRGAN_Compact_Custom_Models

This script will convert a video file using one minute's worth of frames at a time to save disk space. The video fragments are then merged at the end of the process into a final video file.

# You must have ffmpeg installed and a vulkan compatible GPU

Without a GPU this process would take weeks to process a 2 hour video.

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

pip install pillow

git clone https://github.com/davlee1972/upscale_video.git

# Usage

**Step 1 - Calibrate your GPU(s)**

Run python test_gpus.py which will show you what gpus are available.

Run python test_gpus.py -g 0 -s 2 to test gpu 0 running x2 scaling.

Run python test_gpus.py -g 0,0 -s 2 to test gpu 0 using two workers in parallel.

Run python test_gpus.py -g 0,0,0 -s 2 to test gpu 0 using three workers in parallel.

Review timing logs to see how many workers can your GPU support before performance starts to degrade.

**If you have a second, third, etc. gpu, repeat the process for the other gpus.**

Run python test_gpus.py -g 1 -s 2 to test gpu 1 running x2 scaling.

Run python test_gpus.py -g 1,1 -s 2 to test gpu 1 using two workers in parallel.

Run python test_gpus.py -g 1,1,1 -s 2 to test gpu 1 using three workers in parallel.

**Test your final GPU(s) configuration..**

Run python test_gpus.py -g 0,0,0,0,1,1 -s 2 to test gpu 0 with 4 workers and gpu 1 with 2 workers in parallel.

-g, --gpus can be passed into upscale_video.py. If omitted upscale_video.py will just default to -g 0.

**Step 2 - Sample your video upscaling if you want to apply any model filters.**

This is a nice to have if you have older videos and want to remove grain, etc..

You can skip this step if you just want to apply AI upscaling only.

Run upscale_video.py with the -x, --extract_only option.

This will stop processing after frames extraction. 

Run test_images.py on some extracted png files to sample what denoise level (-n, --denoise) to apply if needed. (-a, --anime) can also be passed in to apply anime enhancements.

Run upscale_video.py with -r, --resume_processing to continue with upscaling.

```console
Usage: python upscale_video.py -i INPUT_FILE -f FFMPEG

options:
  -h, --help                            Show this help message and exit
  -i, --input_file INPUT_FILE           Input video file.
  -o, --output_file OUTPUT_FILE         Optional output video file location.
                                         Default is input_file + ('.2x.' or '.4x.')
  -f, --ffmpeg FFMPEG                   Location of ffmpeg.
  -e, --ffmpeg_encoder FFMPEG_ENCODER   ffmpeg encoder for mkv file. Default is av1_qsv.
  -a, --anime                           Adds processing for anime to remove grain and color bleeding.
  -n, --denoise DENOISE                 Adds processing to remove film grain. Denoise level 1 to 30.
                                         3 = light / 10 = heavy, etc.
  -s, --scale SCALE                     Scale 2 or 4. Default is 2.
  -t, --temp_dir TEMP_DIR               Temp directory. Default is tempfile.gettempdir().
  -g
  -b, --batch_size BATCH_SIZE           Number of minutes to upscale per batch. Default is 1.
  -r, --resume_processing               Does not purge any data in temp_dir when restarting.
  -x, --extract_only                    Exits after frames extraction. You may want to
                                         run test_image.py on some extracted png files
                                         to sample what denoise level to apply if needed.
                                         Rerun with -r / --resume_processing to restart.
  -l, --log_level LOG_LEVEL             Logging level. logging.INFO is default
  -d, --log_dir LOG_DIR                 Logging directory. logging directory

```

```console
Usage: python test_images.py -i infile

options:
  -h, --help                          Show this help message and exit
  -i, --input_frames INPUT_FRAMES     Input frames
  -t, --temp_dir TEMP_DIR             Temp directory where extracted frames are saved. Default is tempfile.gettempdir().
  -o, --output_dir OUTPUT_DIR         Output directory where test images will be saved.
                                      Default is input_file + ('.2x.' or '.4x.')
  -s, --scale SCALE                   Scale 2 or 4. Default is 2.
  -a, --anime                         Adds processing for anime to remove grain and smooth color.
  -n, --denoise DENOISE               Adds processing to reduce image grain. Denoise level 1 to 30.
                                      3 = light / 10 = heavy, etc.
  -g, --gpus GPUS                     Optional gpu #s to use. Example 0,1,3. Default is 0.

```

```console
Usage: python test_gpus.py -g gpu

options:
  -h, --help        Show this help message and exit
  -g, --gpus GPUS   Optional gpus to test. examples: 0 or 0,1 or 0,1,1 (to test same gpu twice)

```


# Samples

![alt text](https://i.imgur.com/nkbA0Ft.png)
Original 1920 x 800 extracted image from Underworld Blu-ray

![alt text](https://i.imgur.com/Z2djqQN.png)
Upscaled 2x using --scale 2. Took 40 hours to process 200,000+ frames.

![alt text](https://i.imgur.com/GOFMK47.png)
Upscaled 2x with light denoise using --scale 2 --denoise 3. Denoise added an additional 3 hours of processing.

![alt text](https://i.imgur.com/xG9kwMJ.png)
Original 1920 x 800 extracted image from 2 Fast 2 Furious Blu-ray

![alt text](https://i.imgur.com/dIWhovG.png)
Upscaled 2x with light denoise using --scale 2 --denoise 2.

# Notes

This python code is used to scale my existing 2k bluray collecion to 4k

Troubleshooting Ubuntu. It took me a while to figure out why my AMD gpu on my linux server wasn't working
with Vulkan. Apparently you have to be a member of the ‘render’ group while running in a shell session.

In order to access GPU capabilities, a user needs to have the correct permissions on the system. The following
will list the group assigned ownership of the render nodes, and list the groups the active user is a member of:

```console
stat -c "%G" /dev/dri/render*
groups ${USER}
```

If a group is listed for the render node which isn’t listed for the user, you will need to add
the user to the group using gpasswd. In the following, the active user will be added to the
‘render’ group and a new shell spawned with that group active:

```console
sudo gpasswd -a ${USER} render
newgrp render
```
