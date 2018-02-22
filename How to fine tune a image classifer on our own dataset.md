# How to fine tune a image classifer on our own dataset 

### step1: build image data

*Converts image data to TFRecords file format with Example protos*
- [x] example
- [ ] example

The image data set is expected to reside in JPEG files located in the following directory structure.
  * data_dir/label_0/image0.jpeg
  * data_dir/label_0/image1.jpg
  * ...
  * data_dir/label_1/weird-image.jpeg
  * data_dir/label_1/my-image.jpeg

where the sub-directory is the unique label associated with these images.

This TensorFlow provide script [build_image_data.py](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py)converts the training and evaluation data into a sharded data set consisting of TFRecord files
  * train_directory/train-00000-of-01024
  * train_directory/train-00001-of-01024
  * ...
  * train_directory/train-01023-of-01024
  
and

  * validation_directory/validation-00000-of-00128
  * validation_directory/validation-00001-of-00128
  * ...
  * validation_directory/validation-00127-of-00128
  
where we have selected 1024 and 128 shards for each data set. Each record within the TFRecord file is a serialized Example proto. The Example proto contains the following fields:

  > image/encoded: string containing JPEG encoded image in RGB colorspace
  > image/height: integer, image height in pixels
  > image/width: integer, image width in pixels
  > image/colorspace: string, specifying the colorspace, always 'RGB'
  > image/channels: integer, specifying the number of channels, always 3
  > image/format: string, specifying the format, always 'JPEG'
  > image/filename: string containing the basename of the image file e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  > image/class/label: integer specifying the index in a classification layer. The label ranges from [0, num_labels] where 0 is unused and left as the background class.
  > image/class/text: string specifying the human-readable version of the label e.g. 'dog'
  
If your data set involves bounding boxes, please look at [build_imagenet_data.py]().
