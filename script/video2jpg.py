import os
import subprocess
from moviepy.video.io.VideoFileClip import VideoFileClip
import datetime

def main():
    input_dir = "../dataset/vevo/"
    output_dir = "../dataset/vevo_frame/"
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Loop over all MP4 files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_dir, filename)

            output_dir2 = os.path.join(output_dir, f"{filename[:-4]}")
            if not os.path.exists(output_dir2):
                os.makedirs(output_dir2)
            
            output_path = os.path.join(output_dir2, f"{filename[:-4]}_%03d.jpg")
            cmd = f"ffmpeg -i {input_path} -vf \"select=bitor(gte(t-prev_selected_t\,1)\,isnan(prev_selected_t))\" -vsync 0 -qmin 1 -q:v 1 {output_path}"
            
            subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    main()
