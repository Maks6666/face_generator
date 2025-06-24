import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding)
        self.bnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.bnorm(self.conv(x)))

# --------------------------------------------------------------------------------------------------------------------

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output=False):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding)
        self.bnorm = nn.BatchNorm2d(out_channels)

        self.output = output

    def forward(self, x):
        if self.output == False:
            return F.leaky_relu(self.bnorm(self.conv(x)))
        elif self.output == True:
            return torch.sigmoid(self.bnorm(self.conv(x)))


# --------------------------------------------------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.bnorm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               stride=1, padding=1)

        self.bnorm2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.add_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels)
            )

        else:
            self.add_conv = nn.Identity()

    def forward(self, x):
        out = self.bnorm1(self.conv1(x))
        add_out = self.add_conv(x)

        out = F.leaky_relu(out)

        out = self.bnorm2(self.conv2(out))

        # if add_out is not None:
        out += add_out
        out = F.leaky_relu(out)

        return out

# --------------------------------------------------------------------------------------------------------------------

class UpResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.up_conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, stride=stride, padding=padding)
        self.bnorm1 = nn.BatchNorm2d(out_channels)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels,
                                           kernel_size=3, stride=1, padding=1)
        self.bnorm2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:

            self.add_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels)

            )
        else:
            self.add_conv = nn.Identity()

    def forward(self, x):
        out = self.bnorm1(self.up_conv1(x))
        add_out = self.add_conv(x)

        out = F.leaky_relu(out)

        out = self.bnorm2(self.up_conv2(out))

        out += add_out

        out = F.leaky_relu(out)

        return out

# --------------------------------------------------------------------------------------------------------------------


class SEBlock(nn.Module):
  def __init__(self, C, r=16):
    super().__init__()

    self.aap = nn.AdaptiveAvgPool2d((1, 1))
    self.flatten = nn.Flatten()

    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

    self.linear1 = nn.Linear(C, C//r)
    self.linear2 = nn.Linear(C//r, C)


  def forward(self, x):
    out = self.aap(x)
    out = self.flatten(out)

    out = self.relu(self.linear1(out))
    out = self.sigmoid(self.linear2(out))

    out = out[:, :, None, None]

    res = x * out

    return res

# --------------------------------------------------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, cond=32, latent_dim=128):
        super().__init__()

        self.conv1 = ConvBlock(3, 64, 4, 2, 1)

        self.conv2 = ResBlock(64, 64, 4, 2, 1)

        self.conv3 = ConvBlock(64, 128, 4, 2, 1)

        self.conv4 = ResBlock(128, 128, 4, 2, 1)

        self.conv5 = ConvBlock(128, 256, 4, 2, 1)

        self.conv6 = ConvBlock(256, 512, 4, 2, 1)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(512 * 2 * 2 + cond, 1024)

        self.linear2_mu = nn.Linear(1024, latent_dim)

        self.linear2_logvar = nn.Linear(1024, latent_dim)

    def forward(self, x, y):
        out = self.conv1(x)
        # print(x.shape)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        out = self.flatten(out)

        out = torch.cat((out, y), dim=1)

        h = F.leaky_relu(self.linear1(out))
        # h = F.leaky_relu(self.linear2(out))

        mu = self.linear2_mu(h)
        logvar = self.linear2_logvar(h)
        return mu, logvar

# --------------------------------------------------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, cond=32, latent_dim=128):
        super().__init__()

        self.linear1 = nn.Linear(latent_dim + cond, 1024)
        self.linear2 = nn.Linear(1024, 512 * 2 * 2)
        # x.view(256, 7, 7)
        self.unflatten = nn.Unflatten(1, (512, 2, 2))


        self.t_conv1 = UpResBlock(512, 256, 4, 2, 1)
        self.se_block1 = SEBlock(256)

        self.t_conv2 = UpResBlock(256, 128, 4, 2, 1)
        self.se_block2 = SEBlock(128)

        self.t_conv3 = UpConvBlock(128, 64, 4, 2, 1)
        # self.se_block3 = SEBlock(64)

        self.t_conv4 = UpConvBlock(64, 64, 4, 2, 1)

        self.t_conv5 = UpConvBlock(64, 32, 4, 2, 1)
        # 32, 64, 64
        self.t_conv6 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        # 3, 128, 128

        # self.upsmaple1 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, z, y):
        zy = torch.cat((z, y), dim=1)

        out = F.leaky_relu(self.linear1(zy))
        out = F.leaky_relu(self.linear2(out))

        out = self.unflatten(out)
        out = self.t_conv1(out)
        out = self.se_block1(out)

        out = self.t_conv2(out)
        out = self.se_block2(out)

        out = self.t_conv3(out)
        # out = self.se_block3(out)

        out = self.t_conv4(out)

        out = self.t_conv5(out)

        rec = F.sigmoid(self.t_conv6(out))

        return rec

# --------------------------------------------------------------------------------------------------------------------


class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        # z = μ+σ*ε
        z = mu + std * eps
        return z

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparametrize(mu, logvar)
        rec = self.decoder(z, y)
        return rec, mu, logvar

    def predict(self, x, y):
        self.eval()

        with torch.no_grad():
            if len(x.shape) < 4:
                x = x.unsqueeze(0)
            out, mu, logvar = self.forward(x, y)

            # if len(out.shape) == 3:
            # out = out.squeeze(0)
            out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            #     out = out.squeeze(0)
            # out = out.view(-1, 28, 28)

        return out

    def sample(self, z, y):
        self.eval()

        with torch.no_grad():
            out = self.decoder(z, y)
        out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        return out

# --------------------------------------------------------------------------------------------------------------------

model = VAE()
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"Device: {device}")

model_weights = "/Users/maxkucher/face_generator/models/face_generator_VAE_10.pt"
model.load_state_dict(torch.load(model_weights, map_location=device))
print("Model loaded.")


