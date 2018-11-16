# -*- encoding: utf8 -*-
import pickle
import numpy
import random
import platform
import cv2

class Corpus:

    #类的构造函数
    def __init__(self):
        self.load_cifar10('data/CIFAR10_data')#注意这个路径是以项目的根目录为基准
        self._split_train_valid(valid_rate=0.9)
        self.n_train = self.train_images.shape[0]#训练集数组长度
        self.n_valid = self.valid_images.shape[0]#验证集数组长度
        self.n_test = self.test_images.shape[0] #测试集数组长度

    #获取训练数据和测试数据
    #self 代表类的实例，self 在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数。
    def load_cifar10(self,directory):
        # 读取训练集
        images,labels=[],[]
        # range（start， end， scan)：
        # 参数含义：
        # start:计数从start开始。默认是从0开始。例如range（5）等价于range（0， 5）;
        # end:技术到end结束，但不包括end.例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5
        # scan：每次跳跃的间距，默认为1。例如：range（0， 5） 等价于 range(0, 5, 1)
        for filename in ['%s/data_batch_%d' % (directory,j)for j in range(1,6)]:
            with open(filename,'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10 = pickle.load(fo,encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10 = pickle.load(fo,encoding='bytes')
            for i in range(len(cifar10[b'labels'])):
                # cifar10是一个字典，后面的写法是通过key获取value的写法
                # 每一批cifar10中有10000个训练数据，所以len的值为10000
                # 由于是按顺序存储的，所以每一个标签所对应的value代表数据标签
                # data所对应的图像的分类值。
                image = numpy.reshape(cifar10[b'data'][i],(3,32,32))
                # 提取每一张图片，并进行reshape成(3,32,32)这样的结构，表示rgb三色，每个都是32×32
                image = numpy.transpose(image, (1, 2, 0))
                # 把数据的顺序进行调整，让rgb变成第三个维度
                image = image.astype(float)
                # 转变数据类型为float
                images.append(image)
            labels += cifar10[b"labels"]
        images = numpy.array(images, dtype='float')
        labels = numpy.array(labels, dtype='int')
        self.train_images, self.train_labels = images, labels
        #将读取的结果存储到类的变量train_images、train_labels中

        # 读取测试集
        images, labels = [], []
        for filename in ['%s/test_batch' % (directory)]:
            with open(filename, 'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10 = pickle.load(fo,encoding='bytes')
            for i in range(len(cifar10[b"labels"])):
                image = numpy.reshape(cifar10[b"data"][i], (3, 32, 32))
                image = numpy.transpose(image, (1, 2, 0))
                image = image.astype(float)
                images.append(image)
            labels += cifar10[b"labels"]
        images = numpy.array(images, dtype='float')
        labels = numpy.array(labels, dtype='int')
        self.test_images, self.test_labels = images, labels

    def _split_train_valid(self,valid_rate=0.9):
        # 划分训练集与验证集，valid_rate表示将总数据的90%作为训练集，10%作为验证集
        images, labels = self.train_images, self.train_labels
        thresh = int(images.shape[0] * valid_rate)
        #images.shape[0]表示images数组第一维数据的长度，这里的长度为50000,也就是有这么多张图片

        # 训练集的长度是从开始到thresh阈值所在的位置
        self.train_images, self.train_labels = images[0:thresh, :, :, :], labels[0:thresh]
        # [0:thresh, :, :, :]这种数组的表示方法：用逗号分割不同的维度，这里有三个逗号，说明是四维数组
        # 冒号分割每一维的起始与终止位置，0:thresh表示该维从0开始到thresh结束，冒号前不写数字表示从头开始
        # 冒号后不写数字表示到该维数组的结束位置。

        #验证集长度从thresh开始到数组结束
        self.valid_images, self.valid_labels = images[thresh:, :, :, :], labels[thresh:]

    def data_augmentation(self, images, mode='train', flip=False,
                          crop=False, crop_shape=(24, 24, 3), whiten=False,
                          noise=False, noise_mean=0, noise_std=0.01):
        # 图像切割
        if crop:
            if mode == 'train':
                images = self._image_crop(images, shape=crop_shape)
            elif mode == 'test':
                images = self._image_crop_test(images, shape=crop_shape)
        # 图像翻转
        if flip:
            images = self._image_flip(images)
        # 图像白化
        if whiten:
            images = self._image_whitening(images)
        # 图像噪声
        if noise:
            images = self._image_noise(images, mean=noise_mean, std=noise_std)

        return images

    def _image_crop(self, images, shape):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            old_image = numpy.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
            left = numpy.random.randint(old_image.shape[0] - shape[0] + 1)
            top = numpy.random.randint(old_image.shape[1] - shape[1] + 1)
            new_image = old_image[left: left + shape[0], top: top + shape[1], :]
            new_images.append(new_image)

        return numpy.array(new_images)

    def _image_crop_test(self, images, shape):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            old_image = numpy.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
            left = int((old_image.shape[0] - shape[0]) / 2)
            top = int((old_image.shape[1] - shape[1]) / 2)
            new_image = old_image[left: left + shape[0], top: top + shape[1], :]
            new_images.append(new_image)

        return numpy.array(new_images)

    def _image_flip(self, images):
        # 图像翻转
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            if numpy.random.random() < 0.5:
                new_image = cv2.flip(old_image, 1)
            else:
                new_image = old_image
            images[i, :, :, :] = new_image

        return images

    def _image_whitening(self, images):
        # 图像白化
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            new_image = (old_image - numpy.mean(old_image)) / numpy.std(old_image)
            images[i, :, :, :] = new_image

        return images

    def _image_noise(self, images, mean=0, std=0.01):
        # 图像噪声
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            new_image = old_image
            for i in range(images.shape[0]):
                for j in range(images.shape[1]):
                    for k in range(images.shape[2]):
                        new_image[i, j, k] += random.gauss(mean, std)
            images[i, :, :, :] = new_image

        return images
