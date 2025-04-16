'''
基于Tensorflow2，TF-Hub开源项目——神经网络风格迁移
'''

# 导入和配置模块
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import os
import time

# 张量转化为图像，保存图片
def tensor_to_image(tensor, str):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  # 保存图片
  img = tf.image.encode_png(tensor)
  with tf.io.gfile.GFile(r"D:\Captcha\dataset\captcha_3000_after_conversion\TF\style_cezanne\\"+str+'.jpg', 'wb') as file:
    file.write(img.numpy())
  return PIL.Image.fromarray(tensor)

# 定义一个加载图像的函数，并将其最大尺寸限制为 512 像素
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


# 使用 TF-Hub 进行快速风格迁移
#hub_module = hub.load('./module/magenta_arbitrary-image-stylization-v1-256_1')
hub_module = hub.load('./module/magenta_arbitrary-image-stylization-v1-256_2')
path = r"D:\Captcha\dataset\captcha_3000" #文件夹目录
files= os.listdir(path)
style_image = load_img(r"./style_picture/cezanne1.jpg")
for file in files: #遍历文件夹
  content_image = load_img(path+"\\"+file)
  stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
  name = file[:-4]
  tensor_to_image(stylized_image, name)

