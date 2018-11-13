# -*- coding: utf8 -*-
# author: zerg
import os
from data.corpus import Corpus


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cifar10 = Corpus()

#测试从数据集中导入数据是否成功，显示出来表示成功
def testPic():
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    im = Image.fromarray(np.uint8(cifar10.train_images[0]))
    #cifar10数据集这里是float类型，要输出显示，则要修改为uint8这种类型
    plt.imshow(im)  # 将图片输出
    plt.show()

testPic()