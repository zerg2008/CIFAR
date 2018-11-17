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
def basic_cnn():
    from model.basic_cnn import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, network_path='config/networks/basic.yaml')
    #convnet.debug()
    convnet.train(dataloader=cifar10, backup_path='backups/cifar10-v1/', batch_size=128, n_epoch=20)
    #convnet.test(dataloader=cifar10, backup_path='backups/cifar10-v1/', epoch=0, batch_size=128)
    #convnet.observe_salience(dataloader=cifar10,batch_size=1, n_channel=3, num_test=10, epoch=0)
    #convnet.observe_hidden_distribution(dataloader=cifar10,batch_size=128, n_channel=3, num_test=1, epoch=21)
basic_cnn()