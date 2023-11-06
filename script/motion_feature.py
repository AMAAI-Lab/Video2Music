import os
import math
import cv2

def main():
    directory = "../dataset/vevo_chord/lab/all/"
    directory_vevo = "../dataset/vevo/"
    datadict = {}

    for filename in sorted(os.listdir(directory)):
        print(filename)
        fname = filename.split(".")[0]
        videopath = os.path.join(directory_vevo, filename.replace("lab", "mp4"))
        cap = cv2.VideoCapture(videopath)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        time_incr = 1 / fps
        prev_frame = None
        prev_time = 0
        motion_value = 0
        sec = 0
        motiondict = {}

        while cap.isOpened():
            # Read the frame and get its time stamp
            ret, frame = cap.read()
            if not ret:
                break
            curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Calculate the RGB difference between consecutive frames per second
            motiondict[0] = "0.0000"

            if prev_frame is not None and curr_time - prev_time >= 1:
                diff = cv2.absdiff(frame, prev_frame)
                diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
                motion_value = diff_rgb.mean()
                #print("Motion value for second {}: {}".format(int(curr_time), motion_value))
                motion_value = format(motion_value, ".4f")
                motiondict[int(curr_time)] = str(motion_value)
                prev_time = int(curr_time)
            # Update the variables
            prev_frame = frame.copy()
        # Release the video file and close all windows
        cap.release()
        cv2.destroyAllWindows()

        fpathname = "../dataset/vevo_motion/all/" + fname + ".lab"
        with open(fpathname,'w',encoding = 'utf-8') as f:
            for i in range(0, len(motiondict)):
                f.write(str(i) + " "+motiondict[i]+"\n")

if __name__ == "__main__":
    main()

