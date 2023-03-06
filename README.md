# PyTorch Geography Segmentation Project

This project attempts to develop a Convolutional Neural Network with PyTorch with inspiration taken from the popular UNet encoder-decoder network architecture designed to assist with instance segmentation originally designed to improve medical imaging techniques for the detection of cancerous tumors!

This project builds a custom UNet architecture from scratch using PyTorch and OpenCV and implements multi-class instance segmentation on the Landcover.ai dataset of satellite images collected over Poland with masks provided for 4 class labels - forestry, roads, buildings, farmland. The Landcover.ai dataset can be accessed here: [Landcover.ai homepage](https://landcover.ai.linuxpolska.com/). The following diagram describes the implemented neural network: 


<p align="center"><img src="https://github.com/shlok191/PyTorch_Terrain_Segmentation/blob/main/data/unet-description/u-net-architecture.png" width="50%"></p>

The UNet deep learning structure first downscales all input features into smaller sizes before upscaling all features and converting given input channels to the required output channels (each channel representing one class label's pixels!) in order to accurately classify each pixel as a class unique class object. The process is accomplished by 2 implemented `nn.moduleLists` representing the curves of the "U" structure. Each moduleList consists of 2 custom implemented `DoubleConv` convolution class objects and 1 pooling operation each stage to get desired numbers of channels at each stage.

The following output images were obtained after 3 epochs of training!

<p align="center"><img src="https://github.com/shlok191/PyTorch_Terrain_Segmentation/blob/main/data/results.png" width="50%"></p>

The observed model had an accuracy of 92.2%. Due to a lack of newer GPUs in my tech stack, I did not pursue to train the model beyond 3 epochs. However, I am quite confident that increasing the epochs to ~ 40 would show a drastic improvevemnt in model accuracy!
