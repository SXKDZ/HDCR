import os
import struct
import numpy as np
import matplotlib.pyplot as plt


class DataSet(object):
    def __init__(self, train_filename, label_filename,
                 output_image_path='', output_label_path=''):
        self._train_filename = train_filename
        self._label_filename = label_filename
        self._output_image_path = output_image_path
        self._output_label_path = output_label_path

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
        for i in range(num_images):
            image_value = struct.unpack_from(image_big_endian,
                                             f.read(image_size))
            image_value = list(image_value)
            yield image_value
        f.close()

    def get_image(self):
        i = 1
        for image in self.image_producer():
            img = np.array(image)
            img = img.reshape(28, 28)
            output_file = os.path.join(self._output_image_path, 'image_%d.png' % i)
            plt.figure(frameon=False)
            plt.imshow(img, cmap='binary')  # show the image in BW mode
            plt.axis('off')
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
            plt.close()
            i += 1

    def get_label(self):
        f_in = open(self._label_filename, 'rb')
        f_out = open(os.path.join(self._output_label_path, 'label.txt'), 'w')
        # label header
        header = 'I' * 2
        header_big_endian = '>' + header
        header_size = struct.calcsize(header)
        num_magic, num_items = struct.unpack_from(header_big_endian,
                                                  f_in.read(header_size))
        # label
        label = 'B'
        label_big_endian = '>' + label
        label_size = struct.calcsize(label)
        for i in range(1, num_items + 1):
            label_value = struct.unpack_from(label_big_endian,
                                             f_in.read(label_size))
            label_value = label_value[0]
            f_out.write('%d: %d\n' % (i, label_value))
        f_in.close()
        f_out.close()


dataset = DataSet('images', 'labels')
dataset.get_label()
