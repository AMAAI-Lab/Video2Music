import os
import math
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.scene_manager import save_images

def main():
    directory = "../dataset/vevo_chord/lab/all/"
    directory_vevo = "../dataset/vevo/"
    datadict = {}

    for filename in sorted(os.listdir(directory)):
        print(filename)
        fname = filename.split(".")[0]
        #fname = int(fname)
        videopath = os.path.join(directory_vevo, filename.replace("lab", "mp4"))
        video = open_video(videopath)
        
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector())
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        sec = 0
        scenedict = {}
        for idx, scene in enumerate(scene_list):
            end_int = math.ceil(scene[1].get_seconds())
            for s in range (sec, end_int):
                scenedict[s] = str(idx)
                sec += 1
        
        fpathname = "../dataset/vevo_scene/" + fname + ".lab"
        with open(fpathname,'w',encoding = 'utf-8') as f:
            for i in range(0, len(scenedict)):
                f.write(str(i) + " "+scenedict[i]+"\n")

        
if __name__ == "__main__":
    main()
