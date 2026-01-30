[![Librelane Digital Flow (UNIC-CASS)](https://github.com/unic-cass/unic-cass-wrapper/actions/workflows/digital-flow.yaml/badge.svg?branch=dev&event=push)](https://github.com/unic-cass/unic-cass-wrapper/actions/workflows/digital-flow.yaml)

# UNIC-CASS-WRAPPER 

# CNN4IC
Convolutional Neural Network (CNN) for Image Classification

_Project still in development_

#### Created by:

- Jacobo Morales Erazo
- Martín Calderón
- Hernando Diaz
- Daniel Pedraza

**Chapter/Section:** CASS Universidad de los Andes Student Chapter / Colombia Section

### Description of the Design Idea:

A Convolutional Neural Network (CNN) where weights are changeable leverages the ability for offline and extra-low power classification, a commonly complex and memory intensive process. This IC will be built following the principles of convolutional neural networks, involving the convolutional and pooling layers that make these architectures strong image classifiers, while enabling the option of modified via an external connection to adjust the kernel's weights and, in this way, the CNN can be easily adapted to any context. For training the CNN for handwritten number recognition, the chosen dataset is MNIST, which is a subset of the bigger NIST dataset, which includes 60000 samples for training and 10000 samples for tests. During preprocessing, all the images were changed from gray scale into binary scale. The images will come preprocessed from an external device connected with the chip via an on-chip Serial Peripheral Protocol (SPI) that will manage input and output on information from the chip to the external devices. This serial protocol will keep the number of pins in a manageable quantity and simplifies communication while leaving room for the creation of the CNN. Also, the IC will use a little address protocol to dictacte with type of information is entering the IC (image input values or weight parameter values). 

In detail with the CNN, it will receive images in 0 or 1 format representing a black pixel as 0 and a white pixel as 1. Also, the image will be preprocessed to fit 28 x 28 pixels, this making in total 784 bits per image which is between 2^9 and 2^10 bits per image this size was decided for making possible the recognition of better-quality images from the normal creation of these CNN. Thus, the image resolution makes a realistic objective to achieve the CNN and isn’t too big to manage by the SPI module which could cause delay in chip communication. 

The chosen model was based on the LeNet-5 architecture created by Yann LeCun, which was conceived for the classification of the MNIST dataset. 

Input -> Convolution layer 1 (5x5xn)-> Max pooling (2x2)-> Convolution layer 2 (5x5xm)-> Max pooling (2x2)-> Flatten -> Fully connected layer -> Output  

To create a model as compact as possible, the third convolutional layer and the first fully connected layer were dropped due to their heavy influence on increased amounts of parameters. 

Using this as our standard, the following methodology was used for the comparison of the models. Training was carried out using 10,000 images while 2,000 images were used for tests. Using a small-scale Monte Carlo approach, 10 iterations for 10 models were done. Each one was trained on a basis of 10 epochs using categorical cross entropy as our loss function and stochastic gradient descent as our optimizer. ReLu was used as the activation function for all the layers except for the fully connected layer which uses SoftMax. From the LeNet-5 architecture, n = 6 and m = 16, however, these will be the independent variables to evaluate the relationship between the number of parameters and the achieved training and test accuracy. Due to the lack of space to represent the data obtained we will have to jump directly onto the analysis and conclusions, however, this is available at the Design Example run section. The accuracies tend to increase slightly in a linear pattern when bound to bigger amounts of n and m keeping the value over 93%, additionally, the number of parameters grows in the same matter as well.  With this trend and trying to keep the parameters around the 10,000 parameters margin, the values for n and m were kept at n = 6 and m = 16 proving to be balanced. In this sense, the chosen architecture is constructed as follows: 

 Input -> Convolution layer 1 (5x5x6)-> Max pooling (2x2)-> Convolution layer 2 (5x5x16)-> Max pooling (2x2)-> Flatten -> Fully connected layer -> Output  

This architecture achieved a training accuracy of 98.65%, a test accuracy of 98.05% and parameters count of 10,422. 

Max activation function 

Considering the output of the Fully connected layer is a vector of N elements V = \[V_1, V_2, ..., V_N], where N corresponds to the number of labels in the dataset (which in this project equals to nine), it was possible to formulate an activation function f(V_i) for predictions based on the Softmax activation function, defined as: 

Softmax(V_i) = exp(V_i) / Σ_{j=1..N} exp(V_j) 

where exp(x) represents e^(x). 

According to this definition, Softmax maps V into a vector of probabilities that sum to 1. Here the predicted class is obtained by taking the index i that maximizes the softmax(V_i). In other words: Predicted class = argmax_i softmax(V_i). 

Moreover, Σ_{j=1..N} exp(V_j) will be the same constant for every Softmax(V_i). So argmax_i softmax(V_i) = argmax_i{ exp(V_i) / Σ_{j=1..N} exp(V_j)} = argmax_i{ exp(V_i)}. So, it is possible to firstly set f(V_i) as f(V_i)= exp(V_i). 

Finally, since the exponential function is strictly increasing, the predicted class can be equivalently obtained by directly taking the index of the maximum logit, without explicitly computing the softmax. So, argmax_i (exp(V_i))= argmax_i (V_i) . Finally, f(V_i) can be set as the activation function f(V_i)=V_i, where the predicted class is obtained by taking the index i that maximizes V_i. 

Convolution and Pooling Layer  

In the convolution and pooling section of architecture, there are challenges related to memory management and the dynamic range of the mapping tensors. Although the weights are stored in 8-bit integer precision, the results of the first convolution may require up to 14 bits to be represented. If this wider precision were to propagate through the rest of the architecture, it would force all subsequent registers and datapaths to also use 14-bit (and later even 22-bit) widths, significantly increasing resource usage. 

To avoid this, the convolution output is temporarily stored in a dedicated register bank sized to hold the required wider precision. The results are then rescaled and quantized back to 8-bit integers before continuing to the next stage. While this reduces precision, it was considered a better trade-off compared to the additional area and resources that would be consumed by maintaining wider registers across the entire datapath. 

After rescaling, the matrix passes through the first max-pooling stage. The second convolution produces tensors that can reach values requiring up to 22 bits, so the same strategy is applied: results are held in a wider temporary register bank and then rescaled to 8 bits. Finally, the matrices go through the second max-pooling stage, after which the final mapping tensors are ready for the fully connected layer.
