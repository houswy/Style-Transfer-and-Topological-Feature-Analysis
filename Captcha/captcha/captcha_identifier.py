# 导入包
import muggle_ocr
import os

# 初始化；model_type 包含了 ModelType.OCR/ModelType.Captcha 两种
# ModelType.Captcha 可识别4-6位验证码
sdk = muggle_ocr.SDK(model_type=muggle_ocr.ModelType.Captcha)

path = r"D:\Captcha\dataset\captcha_3000_after_conversion\CycleGAN\double\ukiyoe\ukiyoe_cezanne" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
total,total_correct=3000,0 #总验证码数量，验证正确数量

for file in files: #遍历文件夹
     #file = os.path.join(path, file) #变为绝对路径
     if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
          #f = open(path+"\\"+file); #打开文件
          with open(path+"\\"+file,"rb") as f:
              image_binary = f.read()
          text = sdk.predict(image_bytes=image_binary)
          name = file[:-4]  # 去掉后缀名.jpg/.png
          if text==name.lower() :
              #print(text + "  " + name + "\n")
              total_correct = total_correct + 1;
print("total_correct is : " + str(total_correct))
print("The recognition successful rate after style migration is "+ str(total_correct / total * 100) + " percent") #打印结果

