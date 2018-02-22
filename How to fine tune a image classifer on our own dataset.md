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
