# How to conculate size of output image from a model.
1. TOC
{:toc}
## 1) Size of a batch data: 
In deep learning, when you build and training model you need to know the size of input. It could be a single or group of 1 Dimension data (time-series data) or 2 Dimension data (images).
To take andvatage of multicore, multi-processing thread from modern architecture of CPU and GPU. There data is group in a batch. Therefore, our input data have a shape of 
(N, size(data)) where N is the number of data we want to load into the CPU or GPU for processing. 
For example:
We have a black-white data have a size (28, 28). Now you want to group in batch of 8 images for training our model then the size of data input should be (8, 28, 28)
Otherwisem if we have a RGB data with the same size. In computer, a RGB picture will be formated in the shape of (H,W,C) whith H,W,C stand for Heights, Widths and Channels. a RGB image will have
three channels. Therefore a batch of 8 RGB images will be (8,3,28,28).

## 2) Calculate the shape of a image output from a neurol network (NN) model:

A neural network model is formed by a sequence of many kinds of layers such as convolution, activation, pooling, flattlen etc. The shape of images will be changed by the structure of layers.
The factors which modify their shape includes: the original size, kernel size, padding, stride values. 
The example below give an example of the size of image after a squence model for generator.

### Example: 
Assume we have a class for a generator as below:
```python
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (CelebA is rgb, so 3 is our default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_chan=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 8),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples in the batch, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)
   ```
   This generator will create a model with 5 `convTranspose2D` layers. 
To estimate the size of images output from the single layer, we can refer the formular from the [links](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html).

```math
H_{out} = (H_{in} - 1)*stride[0] + dilation[0]*(kernel_size[0] - 1) + output_padding[0] + 1
```

```math
W_{out} = (W_{in} - 1)*stride[1] + dilation[1]*(kernel_size[1] - 1) + output_padding[1] + 1
```

In our case, if we input wit a image in a shape (N,C,1,1). Then the size of image at two last colume will be:

### Size of image

| Layers | (H,W) |
|-|-|
| $1^{st}$ | (3,3)|
| $2^{nd}$ | (7,7)|
| $3^{rd}$ | (15,15)|
| $4^{th}$ | (31,31)|
| $5^{th}$ | (64,64)|
