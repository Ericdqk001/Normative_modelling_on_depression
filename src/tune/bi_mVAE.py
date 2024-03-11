import matplotlib.pyplot as plt  # NOTE: matplotlib is not installed with the library and must be installed separately
import torch
from multiviewae import mcVAE
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

MNIST_1 = datasets.MNIST(
    "./data/MNIST",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
)

data_1 = MNIST_1.train_data[:, :, :14].reshape(-1, 392).float() / 255.0
data_2 = MNIST_1.train_data[:, :, 14:].reshape(-1, 392).float() / 255.0

data_1 = data_1.to(DEVICE)
data_2 = data_2.to(DEVICE)

mcvae = mcVAE(
    # cfg="./examples/config/example_mnist.yaml",
    input_dim=[392, 392],
    z_dim=64,
).to(DEVICE)

mcvae.fit(data_1, data_2, max_epochs=50, batch_size=1000)


# %%

MNIST_1 = datasets.MNIST(
    "./data/MNIST",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

data_test_1 = MNIST_1.test_data[:, :, :14].reshape(-1, 392).float() / 255.0
data_test_2 = MNIST_1.test_data[:, :, 14:].reshape(-1, 392).float() / 255.0


mcvae_latent = mcvae.predict_latents(data_test_1, data_test_2)

mcvae_latent_view1, mcvae_latent_view2 = mcvae_latent[0], mcvae_latent[1]

mcvae_reconstruction = mcvae.predict_reconstruction(data_test_1, data_test_2)

mcvae_reconstruction_view1_latent1 = mcvae_reconstruction[0][
    0
]  # view 1 reconstruction from latent 1
mcvae_reconstruction_view2_latent1 = mcvae_reconstruction[0][
    1
]  # view 2 reconstruction from latent 1

mcvae_reconstruction_view1_latent2 = mcvae_reconstruction[1][
    0
]  # view 1 reconstruction from latent 2
mcvae_reconstruction_view2_latent2 = mcvae_reconstruction[1][1]


# %%


# Reconstruction plots - how well can the VAE do same view reconstruction?

data_sample = data_test_1[20]
# indices: view 1 latent, view 1 decoder, sample 21
pred_sample = mcvae_reconstruction_view1_latent1[20]

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(data_sample.reshape(28, 14))
axarr[1].imshow(pred_sample.reshape(28, 14))
plt.show()
plt.close()

# Reconstruction plots - how well can the VAE do cross view reconstruction?

# indices: view 1 latent, view 2 decoder, sample 21
data_sample = data_test_2[20]
pred_sample = mcvae_reconstruction_view2_latent1[20]

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(data_sample.reshape(28, 14))
axarr[1].imshow(pred_sample.reshape(28, 14))
plt.show()
plt.close()

# %%
