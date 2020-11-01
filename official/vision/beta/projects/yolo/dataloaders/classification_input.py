# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Classification decoder and parser."""
# Import libraries
import tensorflow as tf

from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import preprocess_ops

MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class Decoder(decoder.Decoder):
  """A tf.Example decoder for classification task."""

  def decode(self, serialized_example):
    sample_dict = {
                'image/encoded': tf.io.encode_jpeg(serialized_example['image'], quality=100), 
                'image/class/label': serialized_example['label'], 
                }
    tf.print(tf.image.extract_jpeg_shape(sample_dict['image/encoded']))
    return sample_dict

# class Parser(parser.Parser):
#   """Parser to parse an image and its annotations into a dictionary of tensors."""

#   def __init__(self,
#                output_size,
#                num_classes,
#                aug_rand_hflip=True,
#                dtype='float32'):
#     """Initializes parameters for parsing annotations in the dataset.

#     Args:
#       output_size: `Tenssor` or `list` for [height, width] of output image. The
#         output_size should be divided by the largest feature stride 2^max_level.
#       num_classes: `float`, number of classes.
#       aug_rand_hflip: `bool`, if True, augment training with random
#         horizontal flip.
#       dtype: `str`, cast output image in dtype. It can be 'float32', 'float16',
#         or 'bfloat16'.
#     """
#     self._output_size = output_size
#     self._aug_rand_hflip = aug_rand_hflip
#     self._num_classes = num_classes
#     if dtype == 'float32':
#       self._dtype = tf.float32
#     elif dtype == 'float16':
#       self._dtype = tf.float16
#     elif dtype == 'bfloat16':
#       self._dtype = tf.bfloat16
#     else:
#       raise ValueError('dtype {!r} is not supported!'.format(dtype))

#   def _parse_train_data(self, decoded_tensors):
#     """Parses data for training."""
#     label = tf.cast(decoded_tensors['image/class/label'], dtype=tf.int32)

#     image_bytes = decoded_tensors['image/encoded']
#     image_shape = tf.image.extract_jpeg_shape(image_bytes)

#     # Crops image.
#     # TODO(pengchong): support image format other than JPEG.
#     cropped_image = preprocess_ops.random_crop_image_v2(
#         image_bytes, image_shape)
#     image = tf.cond(
#         tf.reduce_all(tf.equal(tf.shape(cropped_image), image_shape)),
#         lambda: preprocess_ops.center_crop_image_v2(image_bytes, image_shape),
#         lambda: cropped_image)

#     if self._aug_rand_hflip:
#       image = tf.image.random_flip_left_right(image)

#     # Resizes image.
#     image = tf.image.resize(
#         image, self._output_size, method=tf.image.ResizeMethod.BILINEAR)

#     # Normalizes image with mean and std pixel values.
#     image = preprocess_ops.normalize_image(image,
#                                            offset=MEAN_RGB,
#                                            scale=STDDEV_RGB)

#     # Convert image to self._dtype.
#     image = tf.image.convert_image_dtype(image, self._dtype)

#     return image, label

#   def _parse_eval_data(self, decoded_tensors):
#     """Parses data for evaluation."""
#     label = tf.cast(decoded_tensors['image/class/label'], dtype=tf.int32)
#     image_bytes = decoded_tensors['image/encoded']
#     image_shape = tf.image.extract_jpeg_shape(image_bytes)

#     # Center crops and resizes image.
#     image = preprocess_ops.center_crop_image_v2(image_bytes, image_shape)

#     image = tf.image.resize(
#         image, self._output_size, method=tf.image.ResizeMethod.BILINEAR)

#     image = tf.reshape(image, [self._output_size[0], self._output_size[1], 3])

#     # Normalizes image with mean and std pixel values.
#     image = preprocess_ops.normalize_image(image,
#                                            offset=MEAN_RGB,
#                                            scale=STDDEV_RGB)

#     # Convert image to self._dtype.
#     image = tf.image.convert_image_dtype(image, self._dtype)

#     return image, label