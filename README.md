# Gender_Classifer_tensorflow
Train a gender classifer using tensorflow and our own data

## step1: data collection and argumentation 
  * souce data: RAP(40K), PETA(100K), some wrong classification here, need manunally cleaned, suggested that using samll dataset to train a classification model, then using this model to help you classify data. 
  * our collected data: from office, airport, street 

## step2: pretrained model comparison and selection
  * [tensoflow provided classication Pre-trained Models](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
  * according to your application, make tradeoff between accuracy and speed

## step3: training
  * [simple fine-tune actively training, short time and can get a realtively acceptable result]()
  * [Training from scratch, higher accuracy and long time than fine-tune actively learning]()
  * [How to fine tune a image classifier on Tensorflow Flowers Data](https://github.com/walton-wang929/Gender_Classifer_tensorflow/blob/master/How%20to%20fine%20tune%20a%20image%20classifier%20on%20Tensorflow%20Flowers%20Data.md)
  * [How to fine tune a image classifer on our own dataset](https://github.com/walton-wang929/Gender_Classifer_tensorflow/blob/master/How%20to%20fine%20tune%20a%20image%20classifer%20on%20our%20own%20dataset.md)

## step4: test on your seperate data and test on other data, as possible as more generalized

## step5: deployment to mobile or cloud server 





## reference:
1. [image-classify-server](https://github.com/ccd97/image-classify-server)
2. [GenderClassifierCNN](https://github.com/scoliann/GenderClassifierCNN/blob/master/genderClassification.py)
3. [deep-machine-learning/Retrained-InceptionV3](https://github.com/deep-machine-learning/Retrained-InceptionV3)
4. [tensorflow-for-poets-2: TFlite](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite/#0)
5. [tensorflow-for-poets-2: Optimize for Mobile](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/#0)
6. [googlecodelabs/tensorflow-for-poets-2](https://github.com/googlecodelabs/tensorflow-for-poets-2)
7. [tensorflow-for-poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)
