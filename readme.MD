# Face generator / Edictor based on C-VAE
### Model, based on conditional VAE, which is able generate realistic faces and edit real ones according to features, chosen by user.

## Model 

![](https://lilianweng.github.io/posts/2018-08-12-vae/vae-gaussian.png)

This version of application was made with conditional VAE model - type of a usual variational autoencoder, which accepts as input not only an image, but also a vector of features, which contains information about image and objects on it - for example, colour of person's hair, eyes, form of the face and etc. 

Unlike usual autoencoders, VAE encodes information not into a single vector, but into a distribution of features, which is defined with two parameters: μ - center (mean) value of each distribution and σ - standard deviation (or logarithm of the standard deviation), which defines the spread (variance) of each element in the distribution.



