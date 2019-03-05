import numpy as np
import spams
from skimage.color import rgb2hed, hed2rgb
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2 as cv
import copy

class StainAugmentor(object):

    def __init__(self, method='C'):
        self.methods = {}
        for c in method:
            self.methods[c.upper()] = True

    def getAugmentation(self, I):
        """
        Get a list of augmented images

        :param I: image in RGB uint8
        :return: a list of augmented images
        """
        res = np.array(I, dtype=np.uint8)
        res = res.reshape((1,)+I.shape)

        if 'R' in self.methods:
            imgs = [self.rotate90(I, i) for i in range(1,4)]
            res = np.concatenate((res, imgs))

        if 'C' in self.methods:
            res = [self.he_aug(I) for i in range(9)] 
        
        if 'S' in self.methods:
            imgs = [self.scale_img(I, 1+0.03*i) for i in range(1,9)]
            res = np.concatenate((res, imgs))
            
        if 'E' in self.methods:
            imgs = [self.elastic_transform(I, 100+100*i, 2+i) for i in range(9)]
            res = np.concatenate((res, imgs))
            
        if 'H' in self.methods:
            # use a simple method to change contrast and brightness
            imgs = [self.change_contrast_brightness(I, np.random.uniform(0.8,1.2), np.random.uniform(-100,100)) for i in range(9)]
            res = np.concatenate((res, imgs))

        if 'B' in self.methods:
            imgs = [self.blur(I, i*2) for i in range(1,10)]
            res = np.concatenate((res, imgs))

        if 'G' in self.methods:
           imgs = [self.gaussian_noise(I, mean=0, var=0.02 * i) for i in range(1, 10)]
           res = np.concatenate((res, imgs))

        return res

    def rotate90(self, I, n):
        """
        rotate the image by n*90 degrees

        :param I:
        :param n:
        :return: the rotated image
        """
        M = cv.getRotationMatrix2D((I.shape[0]/2, I.shape[1]/2), 90*n, 1.0)
        return cv.warpAffine(I, M, dsize=I.shape[:2])


    def he_aug(self, I):
        hed = rgb2hed(I)
        for i in range(2):
            a = np.random.uniform(0.95, 1.05)
            b = np.random.uniform(-0.05, 0.05)
            hed[:, :, i] = a * hed[:, :, i] + b
        I = hed2rgb(hed)
        I *= 255
        return I

    def scale_img(self, I, ratio=1.1):
        img = cv.resize(I, None, fx=ratio, fy=ratio, interpolation=cv.INTER_CUBIC)
        center = img.shape[0] / 2
        x0 = int(center - I.shape[0] / 2)
        return img[x0:x0+I.shape[0], x0:x0+I.shape[0], :]

    def elastic_transform(self, I, alpha, sigma, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = I.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        distored_image = map_coordinates(I, indices, order=1, mode='reflect')
        return distored_image.reshape(I.shape)

    def change_contrast_brightness(self, I, alpha=1.2, beta=20):
        I = I * alpha + beta
        return np.clip(I, 0, 255)

    def blur(self, I, blur_size=5):
        img = cv.blur(I, (blur_size, blur_size))
        return img

    def gaussian_noise(self, I, mean=0, var=0.1):
        gauss = np.random.normal(mean, var**0.5, I.shape)
        gauss *= 255
        return np.clip(I+gauss, 0, 255)        

if __name__ == '__main__':
    aug = StainAugmentor(method='G')#RCSEHBG
    I = cv.imread('../crop_data/Benign/b001_0.png')
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

    res = aug.getAugmentation(I)
    for idx,img in enumerate(res):
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite('../aug_data/img%d.png' % idx, img)

