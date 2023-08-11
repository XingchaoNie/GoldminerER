import cv2
import os
from PIL import Image
from moviepy.editor import *


def video2data(path, cuts=[], p=2):
    """
    path: mp4文件路径
    cut = [(begin, end), (begin, end), ...]: 视频分段剪切
    p: 定义每秒抽取的图片数目
    """
    _, mp4_name = os.path.split(path)
    mp4_name = mp4_name.split(".")[0]
    mp4 = VideoFileClip(path)
    sub_mp4 = list()
    if not len(cuts):
        sub_mp4.append(mp4)
    for sub_clip in cuts:
        sub_mp4 = mp4.subclip(sub_clip[0], sub_clip[1])
        sub_mp4.write_videofile(os.path.join("./Repo/video/sub_mp4/", mp4_name + f"sub{sub_clip[0]}_{sub_clip[1]}.mp4"))
    mp4.close()

    # 切完,开抽
    img_path = "./Repo/data/img"
    audio_path = "./Repo/data/audio"

    # 音频抽取
    print("抽音...")
    for sub_mp4_name in os.listdir("./Repo/video/sub_mp4"):
        this_mp4_path = os.path.join("./Repo/video/sub_mp4/", sub_mp4_name)
        sub_mp4 = AudioFileClip(this_mp4_path)
        sub_mp4.write_audiofile(os.path.join("./Repo/data/audio/", sub_mp4_name.split(".")[0] + ".wav"))
        sub_mp4.close()
    print("抽音完成")
    
    # 图片抽取
    print(mp4_name, "抽帧...")
    idx = 0
    for sub_mp4 in os.listdir("./Repo/video/sub_mp4"):
        idx += 1
        c = 1
        save_tag = 1
        this_mp4_path = os.path.join("./Repo/video/sub_mp4/", sub_mp4)
        this_capture = cv2.VideoCapture(this_mp4_path)
        fps = this_capture.get(5)
        while True:
            ret, frame = this_capture.read()
            if ret:
                frame_rate = int(fps) // p
                if c % frame_rate == 0:
                    # 抽取
                    cv2.imwrite(os.path.join(r"./Repo/data/img/", sub_mp4.split(".")[0] + f"{idx}_{save_tag}.jpg"), frame)
                    save_tag += 1
                c += 1
                cv2.waitKey(0)
            else:
                print(sub_mp4, "抽帧完成")
                break


def img2gif(p):
    gif_dict = dict()
    for img_name in os.listdir(p.img_results_path):
        keys = img_name.split("_")
        img_path = os.path.join(p.img_results_path, img_name)
        img = cv2.imread(img_path)
        w = img.shape[1]
        h = img.shape[0]
        if str(keys[0]) + str(keys[1]) in gif_dict.keys():
            gif_dict[str(keys[0]) + str(keys[1])].append((int(keys[2].split(".")[0]), img))
        else:
            gif_dict[str(keys[0]) + str(keys[1])] = [(int(keys[2].split(".")[0]), img)]
    
    for gif_key in gif_dict.keys():

        gif = cv2.VideoWriter(os.path.join(p.gif_results_path, str(gif_key) + ".avi"), cv2.VideoWriter_fourcc('I', '4', '2', '0'), 2, (w ,h))

        img_list = gif_dict[gif_key].copy()
        img_list.sort(key=lambda x: x[0])

        for img in [item[1] for item in img_list]:
            gif.write(img)
        
        gif.release()
        cv2.destroyAllWindows()
