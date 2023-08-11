import os
from PIL import Image
import numpy as np
from keras.models import load_model
# 音视频分离及视频抽帧
from utils import video2data, img2gif
# 图像情绪识别部分
from imageR import img2emotion
# 音频情绪识别部分
from audioR import audio2emotion
# 所有模型参数
from params import Params
param = Params()
model = load_model(os.path.join(param.img_model_path, "model_v6_23.hdf5"))


if __name__ == "__main__":
    video_tag = False
    if video_tag:
        print("音视频分离及视频抽帧...")
        video2data(path=os.path.join("./Repo/video/mp4/", "test.mp4"),
                cuts=[(4,15), (37,50), (60,94), (105,140), (153,173)],
                p=2)
    else:
        # 图像情绪识别
        for img in os.listdir(param.img_path):
            # 使用绝对路径传参
            img_path = os.path.join(param.img_path, img)
            img_emotion = img2emotion(img_path, param, model)
        # 图像合并gif
        print("进行图像gif合成...")
        img2gif(param)

