# Face generator / Edictor based on C-VAE
## Model, based on conditional VAE, which is able generate realistic faces and edit real ones according
## to features.


# Face generator / Edictor based on C-VAE
### Model, based on conditional VAE, which is able generate realistic faces and edit real ones according to features, chosen by user.

## Model 

![](https://lilianweng.github.io/posts/2018-08-12-vae/vae-gaussian.png)

This version of application was made with conditional VAE model - type of a usual variational autoencoder, which accepts as input not only an image, but also a vector of features, which contains information about image and objects on it - for example, colour of person's hair, eyes, form of the face and etc. 

Unlike usual autoencoders, VAE encodes information not into a single vector, but into a distribution of features - q(z|x), which is defined with two parameters: μ - center (mean) value of each distribution and σ - standard deviation (or logarithm of the standard deviation), which defines the spread (variance) of each element in the distribution. Then model sample a random vector Z from it, which will be decoded by decoder part of model and which may look like an original input. Z value could be sampled using μ and σ values with bext formula:

```
z = μ + μ * e
```

Where e - is a standart normal distribution N(0, I).

## Specific of training

Training of VAE requires complicated sort of a loss function, which contains reconstruction loss (MSE or BCE) to define a differance between input and output image and KL-divergence to define differance between two distributions - q(z|x), into which VAE encodes an image and normal distribution N(0, I). This way we make sampled value Z look possibly same as N(0, I) distribution.