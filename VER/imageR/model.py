import face_recognition
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont


def img2emotion(img_path, p, model):
    emotion_dict= {'愤怒': 0, '悲伤': 5, '中性': 4, '厌恶': 1, '惊喜': 6, '害怕': 2, '高兴': 3}
    img = cv2.imread(img_path)

    face_locations = face_recognition.face_locations(img)
    if len(face_locations):
        top, right, bottom, left = face_locations[0]
    else:
        return
    face_image = img[top:bottom, left:right]

    face_image = cv2.resize(img, (48,48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    predicted_class = np.argmax(model.predict(face_image))
    label_map = dict((v,k) for k,v in emotion_dict.items()) 
    predicted_label = label_map[predicted_class]


    # 存储结果
    res = cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    res = cv2AddChineseText(res, predicted_label, (right, top-50), (255, 0, 0))

    _, img_name = os.path.split(img_path)
    cv2.imwrite(os.path.join(p.img_results_path, img_name), res)
    return predicted_label


def cv2AddChineseText(img, text, position, textColor=(255, 0, 0), textSize=40):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)