# @Time : 2020/3/26 23:32 
# @Author : Jinguo Zhu
# @File : show.py 
# @Software: PyCharm
'''
                    .::::.
                  .::::::::.
                 :::::::::::  I && YOU
             ..:::::::::::'
           '::::::::::::'
             .::::::::::
        '::::::::::::::..
             ..::::::::::::.
           ``::::::::::::::::
            ::::``:::::::::'        .:::.
           ::::'   ':::::'       .::::::::.
         .::::'      ::::     .:::::::'::::.
        .:::'       :::::  .:::::::::' ':::::.
       .::'        :::::.:::::::::'      ':::::.
      .::'         ::::::::::::::'         ``::::.
  ...:::           ::::::::::::'              ``::.
 ````':.          ':::::::::'                  ::::..
                    '.:::::'                    ':'````..
 '''

import matplotlib.pyplot as plt
import cv2
import numpy as np

img = np.zeros([200,200,3])
img[100:150, :, :] =1
plt.imshow(img)
plt.show()

# 如果要使用 opencv远程显示 需要开启 x11
# 服务器端
# (tf13) lechatelia@amax:/data/DataSets/UCF101$ echo $DISPLAY
# localhost:10.0
# 那么就跟该python脚本的环境变量添加
# DISPLAY localhost:10.0


# cv2.imshow('te0', img)
# cv2.waitKey(0)