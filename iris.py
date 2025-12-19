import numpy as nmp
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data[:5]
print ( X )
Y = iris.target[:5]
print ( Y )

from sklearn.datasets import load_sample_image
china = load_sample_image('china.jpg')
print ( china.shape )
from PIL import Image


'''
Image.fromarray(china).show()
Image.fromarray(china[:,:,0]).show()
Image.fromarray(china[:,:,1]).show()
Image.fromarray(china[:,:,2]).show()
'''

images = nmp.load ( "cifar10_test_images.npy")
print ( images.shape )
print ( images.shape[0] )
for i in range ( images.shape[0] ):
    Image.fromarray(images[i,:,:]).show()