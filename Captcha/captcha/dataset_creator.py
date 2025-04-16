from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string
from PIL import Image

#characters为验证码上的字符集，10个数字加26个大写英文字母
#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ str类型
characters=string.digits+string.ascii_uppercase+string.ascii_lowercase
width,height,n_len,n_class=170,80,4,len(characters)

#生成三千张验证码
for i in range(3000):
    generator=ImageCaptcha(width=width,height=height)
    random_str=''.join([random.choice(characters) for j in range(4)])
    img=generator.generate_image(random_str)

    #将图片保存在目录captcha文件夹下
    file_name=r'D:\\Captcha\\dataset\\captcha_3000\\'+random_str+'.jpg'
    img.save(file_name)