import os
import struct
import threadpool
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog


class DataSet(object):
    def __init__(self, train_filename, label_filename,
                 output_image_path='', output_label_path=''):
        self._train_filename = train_filename
        self._label_filename = label_filename
        self._output_image_path = output_image_path
        self._output_label_path = output_label_path
        self._gen = self.image_producer()

    def image_producer(self):
        f = open(self._train_filename, 'rb')
        # image header
        header = 'I' * 4
        header_big_endian = '>' + header
        header_size = struct.calcsize(header)
        num_magic, num_images, num_rows, num_cols = struct.unpack_from(header_big_endian,
                                                                       f.read(header_size))
        # image
        image = '784B'  # 28 * 28
        image_big_endian = '>' + image
        image_size = struct.calcsize(image)
        for i in range(1, num_images + 1):
            image_value = struct.unpack_from(image_big_endian,
                                             f.read(image_size))
            image_value = list(image_value)
            yield (image_value, i)
        f.close()

    def image_consumer(self):
        (image, i) = next(self._gen)
        print('parsing %d' % i)
        img = np.array(image)
        img = img.reshape(28, 28)
        output_file = os.path.join(self._output_image_path, 'image_%d.png' % i)
        plt.figure(frameon=False)
        plt.imshow(img, cmap='binary')  # show the image in BW mode
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()

    def get_images(self):
        images = []
        for (feature, i) in self.image_producer():
            images.append(feature)
        return np.array(images)

    def get_hog_images(self):
        list_hog_fd = []
        original_features = self.get_images()
        for feature in original_features:
            fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                     visualise=False)
            list_hog_fd.append(fd)
        return np.array(list_hog_fd, 'float64')

    def get_image_files(self):
        pool = threadpool.ThreadPool(10)
        requests = []
        for i in range(60000):
            requests.append(threadpool.makeRequests(self.image_consumer(), []))
        map(pool.putRequest, requests)
        pool.poll()

    def label_producer(self):
        f = open(self._label_filename, 'rb')

        # label header
        header = 'I' * 2
        header_big_endian = '>' + header
        header_size = struct.calcsize(header)
        num_magic, num_items = struct.unpack_from(header_big_endian,
                                                  f.read(header_size))
        # label
        label = 'B'
        label_big_endian = '>' + label
        label_size = struct.calcsize(label)
        for i in range(1, num_items + 1):
            label_value = struct.unpack_from(label_big_endian,
                                             f.read(label_size))
            label_value = label_value[0]
            yield (label_value, i)
        f.close()

    def get_labels(self):
        labels = []
        for (value, i) in self.label_producer():
            labels.append(value)
        return np.array(labels)

    def get_label_file(self):
        f_out = open(os.path.join(self._output_label_path, 'label.txt'), 'w')
        for (value, i) in self.label_producer():
            f_out.write('%d: %d\n' % (i, value))
        f_out.close()
