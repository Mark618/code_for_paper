# SRGAN Implementation in PyTorch

This project contains a PyTorch implementation of **SRGAN (Super-Resolution Generative Adversarial Network)** based on the paper:

> *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*  
> Ledig et al., 2017

The objective of SRGAN is to reconstruct high-resolution (HR) images from low-resolution (LR) inputs while preserving realistic textures and visual quality.

---

# Project Structure

```bash
srgan.ipynb
```

The notebook includes:

* Library imports
* Generator architecture
* Residual blocks
* Upsampling blocks
* Discriminator architecture
* Model testing using random tensors

---

# Features

* Implemented fully in **PyTorch**
* Residual block-based generator
* PixelShuffle upsampling
* GAN-based discriminator
* Modular class-based design
* Supports image super-resolution tasks

---

# Model Architecture

## Generator

The generator is responsible for converting low-resolution images into high-resolution images.

### Components

* Initial convolution layer
* Residual blocks
* Batch normalization
* PReLU activations
* Skip connections
* PixelShuffle upsampling
* Final reconstruction layer

### Output

Produces a super-resolved RGB image.

---

## Discriminator

The discriminator distinguishes between:

* Real high-resolution images
* Generated super-resolution images

### Components

* Convolutional blocks
* Batch normalization
* LeakyReLU activations
* Fully connected classification layers

### Output

A probability score indicating whether the image is real or generated.

# Sample Tensor Test

The notebook includes basic testing using random tensors:

```python
temp_input = torch.rand((1,3,96,96))
output = temp_net(temp_input)
```

This verifies:

* Generator output shape
* Discriminator compatibility
* Successful forward propagation

---

# Reference

Ledig, C., Theis, L., Huszár, F., et al.
**Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network**
[https://arxiv.org/abs/1609.04802](https://arxiv.org/abs/1609.04802)

---
