import os
import os.path as osp

import fire
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


"""From https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/subsets.html"""
EPSILON = np.finfo(np.float32).tiny
class SubsetOperator(torch.nn.Module):
    def __init__(self, k, tau=1.0, hard=False):
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau

    def forward(self, scores):
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).cuda())
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot)
            val, ind = torch.topk(khot, self.k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res

"""Adapted from https://github.com/mtrencseni/pytorch-playground/blob/master/14-pytorch-autoencoder/Pytorch%20MNIST%20Autoencoder.ipynb"""
class Autoencoder(nn.Module):
    def __init__(self, d_latent, latent_k, tau=1.0, sub_for_softmax=False):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.LeakyReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.LeakyReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, d_latent),
            # neurons_to_gen x n_neurons_on
            )
        self.subsetop = nn.Softmax() if latent_k==1 and sub_for_softmax else SubsetOperator(k=latent_k, tau=tau)
        self.decoder = nn.Sequential(
            # neurons_to_gen x n_neurons_on
            nn.Flatten(),
            nn.Linear(d_latent, 400),
            # 400
            nn.LeakyReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.LeakyReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            nn.ReLU(),
            #nn.Sigmoid(),
            )
    def forward(self, x):
        enc = self.encoder(x)
        act = self.subsetop(enc)
        dec = self.decoder(act)
        return dec


def gen_embs(folder, dset, model, latent_k, device, plot_preds=False, batch_size=10*1000, num_workers=4, disable_tqdm=True):
    dataloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    os.makedirs(osp.join(folder), exist_ok=True)
    pred_emb = []
    pred_act = []
    pred_idx = []
    label_idx = []
    j = 0
    with torch.no_grad():
        for data in tqdm(dataloader, disable=disable_tqdm, desc=folder):
            cpu_imgs, labels = data
            imgs = cpu_imgs.to(device)
            enc = model.encoder(imgs)
            act = model.subsetop(enc)
            act_idx_np = np.argsort(enc.detach().cpu().numpy())[:,-latent_k:]
            pred_emb.extend(enc.detach().cpu().numpy())
            pred_act.extend(act.detach().cpu().numpy())
            pred_idx.extend(act_idx_np)
            label_idx.extend(labels)
            if plot_preds:
                act_sparse = torch.zeros_like(act, device=device)
                for i in range(act_idx_np.shape[0]):
                    act_sparse[i,act_idx_np[i]] = 1
                decs = model.decoder(act).detach().cpu().numpy()
                decs_sparse = model.decoder(act_sparse).detach().cpu().numpy()
                for i in range(decs.shape[0]):
                    os.makedirs(osp.join(folder,f"{labels[i]}"), exist_ok=True)
                    plt.imsave(osp.join(folder,f"{labels[i]}", f"{j}_y.png"), cpu_imgs[i,0])
                    plt.imsave(osp.join(folder,f"{labels[i]}", f"{j}_ŷ.png"), decs[i,0])
                    plt.imsave(osp.join(folder,f"{labels[i]}", f"{j}_ỹ.png"), decs_sparse[i,0])
                    j+=1
    label_idx = np.stack(label_idx, axis=0)
    pred_emb = np.stack(pred_emb, axis=0)
    pred_act = np.stack(pred_act, axis=0)
    pred_idx = np.stack(pred_idx, axis=0)
    print(*list(map(lambda x: x.shape, [label_idx, pred_emb, pred_idx])))
    df = pd.DataFrame(data={"label": label_idx, **{f"neuron_{latent_k-i}":pred_idx[:,i] for i in range(latent_k)}})
    df.to_csv(osp.join(folder,"mnist_idx.csv"))
    np.save(osp.join(folder,"pred_emb.npy"), pred_emb)
    np.save(osp.join(folder,"pred_act.npy"), pred_act)
    np.save(osp.join(folder,"pred_idx.npy"), pred_idx)


def main(latent_d = 30, latent_k = 6, tau=1.0, num_epochs=128, batch_size=64, save_train=True, plot_train=False, save_test=True, plot_test=True, train=True):
    """Train an Autoencoder with `latent_d` neurons that are run through a gumbel top-k softmax as their hidden representation, which will end up with overlapping (summed) distributions summing to `latent_k` and can be discretised to `latent_k` numbers."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder(latent_d, latent_k, tau).to(device=device)
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.1307,), (0.3081,))
    ])
    if train:
        distance = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        trainset = tv.datasets.MNIST(root='~/data',  train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        with tqdm(range(num_epochs)) as counter:
            for epoch in counter:
                once = True
                for data in dataloader:
                    img, labels = data
                    if once:
                        os.makedirs(osp.join("log",f"{epoch}"), exist_ok=True)
                        for i in range(img.shape[0]):
                            plt.imsave(osp.join("log",f"{epoch}", f"{labels[i]}_{i}_y.png"), img[i,0])
                    img = img.to(device)
                    output = model(img)
                    loss = distance(output, img)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if once:
                        decs = output.detach().cpu().numpy()
                        once = False
                        for i in range(img.shape[0]):
                            plt.imsave(osp.join("log",f"{epoch}", f"{labels[i]}_{i}_yhat.png"), decs[i,0])
                counter.write('{} {:.4f}'.format(epoch+1, loss.item()))
                counter.set_description('epoch [{}/{}], loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        
        torch.save(model.cpu().state_dict(), "trained.pt")
    else:
        model.load_state_dict(torch.load("trained.pt"))
    model.to(device)
    if save_test or plot_test:
        testset = tv.datasets.MNIST(root='~/data',  train=False, download=True, transform=transform)
        gen_embs("test", testset, model, latent_k, device, plot_preds=plot_test, disable_tqdm=False)
    if save_train or plot_train:
        trainset = tv.datasets.MNIST(root='~/data',  train=True, download=True, transform=transform)
        gen_embs("train", trainset, model, latent_k, device, plot_preds=plot_train, disable_tqdm=False)

if __name__=="__main__":
    fire.Fire(main)
