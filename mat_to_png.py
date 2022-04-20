import pickle
import numpy as np
import h5py
from PIL import Image

if __name__ == '__main__':
    path = '/home/luc/Downloads/SUNRGBDtoolbox/SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat'
    dataroot = '/media/luc/data/'
    f = h5py.File(path,'r')
    data = f.get('SUNRGBD2Dseg/')

    with open('filepaths.pylist') as file:
        filepaths = eval(file.read())

    for image, path in zip(data['seglabelall'],filepaths):
        im = Image.fromarray(f[image[0]][:].transpose())
        im.save(dataroot+path+"/mask.tif")

    print('done')
    # f[image[0]][:]
    # f[data['seglabelall'][0][0]]
    # '/n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/image/0000103.jpg'

