[TensorFlow] Generative Adversarial Nets (GAN)
=====

TensorFlow implementation of Generative Adversarial Networks (GAN) with MNIST dataset.  

## Architecture

### Training algorithm
<div align="center">
  <img src="./figures/algorithm.png" width="500">  
  <p>The algorithm for training GAN [1].</p>
</div>

### GAN architecture
<div align="center">
  <img src="./figures/gan.png" width="500">  
  <p>The architecture of GAN [1].</p>
</div>

### Graph in TensorBoard
<div align="center">
  <img src="./figures/graph.png" width="650">  
  <p>Graph of GAN.</p>
</div>

## Results

### Training Procedure
<div align="center">
  <p>
    <img src="./figures/GAN_loss_d.svg" width="300">
    <img src="./figures/GAN_loss_g.svg" width="300">
  </p>
  <p>Loss graph in the training procedure. </br> Each graph shows loss of the discriminator and loss of the generator respectively.</p>
</div>

### Test Procedure
|z:2|z:2 (latent space walking)|
|:---:|:---:|
|<img src="./figures/z02.png" width="300">|<img src="./figures/z02_lw.png" width="300">|

|z:64|z:128|
|:---:|:---:|
|<img src="./figures/z64.png" width="300">|<img src="./figures/z128.png" width="300">|

## Environment
* Python 3.7.4  
* Tensorflow 1.14.0  
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  


## Reference
[1] Ian Goodfellow et al. (2014). <a href="http://papers.nips.cc/paper/5423-generative-adversarial-nets">Generative Adversarial Nets</a>. NIPS (NeurIPS).  
