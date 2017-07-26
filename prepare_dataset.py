#!/usr/bin/env python
import argparse
import os
import random
import time

import tensorflow as tf

from adain.image import load_image
from adain.util import extract_image_names_recursive


def prepare_dataset(input_dir, output_dir, size, images_per_file, file_prefix):
    assert os.path.exists(input_dir), 'Input directory does not exist'
    assert os.path.isdir(input_dir), '%s is not a directory' % input_dir
    assert os.path.exists(output_dir), 'Output directory does not exist'
    assert os.path.isdir(output_dir), '%s is not a directory' % output_dir

    filenames = extract_image_names_recursive(input_dir)
    random.shuffle(filenames)
    print("%s images found in %s" % (len(filenames), input_dir))

    start = time.time()
    errors = 0
    rate = 0
    update_stat_every = 100
    writer = None
    for i, filename in enumerate(filenames):
        if i % images_per_file == 0: # roll to a new file
            pass
            if writer:
                writer.close()
            output_file = '%s-%04i.tfrecords' % (file_prefix, i // images_per_file)
            output_path = os.path.join(output_dir, output_file)
            writer = tf.python_io.TFRecordWriter(output_path)
        try:
            image = load_image(filename, size, crop=True, normalize=False)
            example = build_example(image)
            writer.write(example.SerializeToString())
        except (OSError, OverflowError, ValueError):
            errors += 1
        print(i, '\t', '%0.4f image/sec, %s errors' % (rate, errors), end='\r')
        if i % update_stat_every == 0:
            rate = i / (time.time() - start)

    print('%s images processed at %0.4f image/sec. %s errors occurred.' %
        (len(filenames), rate, errors))


def build_example(image):
    image = image.tostring()
    return tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
    }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Prepare images for training.
        Each image is resized to the specified size, center-cropped, and
        transposed from HWC to CHW.""")
    parser.add_argument('input_dir', help='Directory containing images (jpg files only)')
    parser.add_argument('output_dir', help='Output directory for TFRecords files')
    parser.add_argument('--size', type=int, default=512,
        help='Scale to this minimum size before cropping, keep original size if set to zero')
    parser.add_argument('--images_per_file', type=int, default=5000,
        help='How many images to have in a single TFRecords file')
    parser.add_argument('--file_prefix', default='train',
        help='A prefix to add to TFRecords files')
    args = parser.parse_args()

    prepare_dataset(**vars(args))
