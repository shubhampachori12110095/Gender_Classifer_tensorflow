# How to Train from Scratch

**WARNING:** Training an Inception v3 network from scratch is a computationally intensive task and depending on your compute setup may take several days or even weeks.

here we just talk about train a task on one GPU, later in future I will write how to train using mutiple GPUs. 

The training speed is dictated by many factors -- most importantly the **batch size** and the **learning rate** schedule. Both of these parameters are heavily coupled to the hardware set up.

Generally speaking, a batch size is a difficult parameter to tune as it requires balancing memory demands of the model, memory available on the GPU and speed of computation. Generally speaking, employing larger batch sizes leads to more efficient computation and potentially more efficient training steps. For different task and different network, the batch size is different. For example, **object detection**, if you use SSD, the general batch size is 32 / 64 ; if choose Faster RCNN, then bacth size is 1. 

**so note that : depending your hardware set up, you need to adapt the batch size and learning rate schedule.**

[Tensorflow provideed example on Flowers classification](https://github.com/tensorflow/models/tree/master/research/inception) 
above provided Linux relization, below is Windows 10 :

## Linux environment :

### step1:Downloading Flowers data and convert to TFrecord
```
# location of where to place the flowers data
FLOWERS_DATA_DIR=/tmp/flowers-data/

# build the preprocessing script.
cd tensorflow-models/inception
bazel build //inception:download_and_preprocess_flowers

# run it
bazel-bin/inception/download_and_preprocess_flowers "${FLOWERS_DATA_DIR}"
```
If the script runs successfully, the final line of the terminal output should look like:

```
2016-02-24 20:42:25.067551: Finished writing all 3170 images in data set.
```
When the script finishes you will find 2 shards for the training and validation files in the DATA_DIR. The files will match the patterns train-?????-of-00002 and validation-?????-of-00002, respectively.

**note: **

If you wish to prepare a custom image data set for transfer learning, you will need to invoke build_image_data.py on your custom data set. Please see the associated options and assumptions behind this script by reading the comments section of build_image_data.py. Also, if your custom data has a different number of examples or classes, you need to change the appropriate values in imagenet_data.py.

### step2: Downloading Inception v3 image model
```
# location of where to place the Inception v3 model
mkdir -p inception-v3-model
cd inception-v3-model

# download the Inception v3 model
curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
tar xzf inception-v3-2016-03-01.tar.gz

# this will create a directory called inception-v3 which contains the following files.
> ls inception-v3
README.txt
checkpoint
model.ckpt-157585
```


