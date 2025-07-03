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
z = μ + σ  * e
```

Where e - is a standart normal distribution N(0, I).

## Specific of training

Training of VAE requires complicated sort of a loss function, which contains reconstruction loss (MSE or BCE) to define a differance between input and output image and KL-divergence to define differance between two distributions - q(z|x), into which VAE encodes an image and normal distribution N(0, I). This way we make sampled value Z look possibly same as N(0, I) distribution.
In both version's of VAE (9th and 10th - see "loss_function"), I used combined loss functions for image reconstruction:

```
BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
MSL = F.mse_loss(recon_x, x)
```
KL-divergence was also calculated this way:

```
KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```
Where 'log_var' is σ - standard deviation value, which is common practice for VAE in case of KL calculation. 
Here I also used VGG-16 network for perceptual loss calculation:

```
PL = perceptual_loss(x, recon_x)
```

As a result, in 9th version, I just added all loss values:
```
total_loss = (BCE + MSL + KLD * kl_weight + PL)
```

But in 10th version, I also calculated weighted values of losses, which also improved result of training:

```
recon_loss = rec_alpha * bce + (1 - rec_alpha) * mse
perceptual_loss = perceptual_lambda * perceptual_loss_fn(x, recon_x)
```
And finally, I summarized weighted results the same way:
```
total_loss = recon_loss + kl_weight * kl_beta * kld + perceptual_loss
```
It was also important to add a KL-weight to regularize role of KL value during training:
```
kl_weight = 1 / (1 + np.exp(-0.1 * (epoch - 5)))  
```
### Note: 
You may also try to use simple loss function to calculate loss between generated data and real, but make sure 
you are using mean squared error instead of binary cross entropy - this one could led to bad quality of 
generated images and to appearance of artifacts on it. 

## Results:


