## convert scene_id to scene_offset

import shutil
import os

def convert_format_id_to_offset(id_list):
    offset_list = []
    current_id = id_list[0]
    offset = 0
    for i in range(len(id_list)):
        if id_list[i] != current_id:
            current_id = id_list[i]
            offset = 0
        offset_list.append(offset)
        offset += 1
    return offset_list

def main():
    pp = "../dataset/vevo_scene/"
    for path, subdirs, files in os.walk( pp ):
        for fname in files:
            src = os.path.join(path,fname)
            tgt = src.replace("vevo_scene", "vevo_scene_offset" )        
            id_list = []
            with open(src, encoding = 'utf-8') as f:
                for line in f:
                    line = line.strip()
                    line_arr = line.split(" ")
                    if len(line_arr) == 2 :
                        time = int(line_arr[0])
                        scene_id = int(line_arr[1])
                        id_list.append(scene_id)
            if len(id_list) == 0:
                print("empty file...")
                print(src)
            else:
                offset_list = convert_format_id_to_offset(id_list)
                with open(tgt,'w',encoding = 'utf-8') as f:
                    for i in range(0, len(offset_list)):
                        f.write(str(i) + " " + str(offset_list[i]) + "\n")

if __name__ == "__main__":
    main()
