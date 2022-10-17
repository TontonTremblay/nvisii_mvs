import subprocess
import glob

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        default=None,
        help = "folder of images"
    )

    parser.add_argument(
        '--fps',
        default = 24,
        type = float,
        help = "video fps"
    )

    parser.add_argument(
        '--outf',
        default = 'video.mp4',
        help = "video output file"
    )

    opt = parser.parse_args()
         


    subprocess.call(['ffmpeg',\
    '-y',\
    '-framerate', str(opt.fps), \
    '-pattern_type', 'glob', '-i',\
    f"{opt.path}/*.png", 
    "-c:v", "libx264","-pix_fmt", "yuv420p",\
    opt.outf
    ]) 