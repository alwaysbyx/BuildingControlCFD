import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class venti_controler:
    def __init__(self, models, train_dataset, g, x, f, device):
        self.freeze_model(models)
        self.y_normalizer = train_dataset.y_normalizer.to(device) 
        self.up_normalizer = train_dataset.up_normalizer.to(device) 
        self.target_co2 =  self.y_normalizer.transform(400.*torch.ones(7492,6).to(device), inverse=False).to(device)
        self.xmin = self.up_normalizer.transform(0.324*torch.ones(1,13).to(device), inverse=False).to(device)
        self.xmax = self.up_normalizer.transform(3.24*torch.ones(1,13).to(device), inverse=False).to(device)
        self.xmin2 = self.up_normalizer.transform(45.*torch.ones(1,13).to(device), inverse=False).to(device)
        self.xmax2 = self.up_normalizer.transform(135.*torch.ones(1,13).to(device), inverse=False).to(device)
        self.criterion = nn.MSELoss()
        self.g = g.to(device)
        self.x0 = x.to(device)
        self.f = f.to(device)
        self.device = device

    def relative_error(self, x_hat, x,  p=2):
        error = torch.norm(x - x_hat, p=p) / torch.norm(x, p=p)
        return error
            
    def freeze_model(self, models):
        torch.manual_seed(0)
        self.models = models
        for model in self.models:
            model.eval()
            for param in model.parameters(): param.requires_grad = False

    def get_prediction(self, g, cnt, f):
        if len(self.models) == 1:
            model = self.models[0]
            mu, sigma = model(g, cnt, f)
            std = torch.sqrt(F.softplus(sigma) + 1e-6)
        else:
            mus, sigmas = [], []
            for model in self.models:
                mu, sigma = model(g, cnt, f)
                mus.append(mu)
                sigmas.append(F.softplus(sigma) + mu ** 2)
            mu = torch.stack(mus).mean(dim=0) 
            sigma = torch.stack(sigmas).mean(dim=0) - mu**2
            std = torch.sqrt(sigma)
        return mu, std
    

    def get_air_penalty(self, g, cnt, f):
        """
        Input: the past 12 data, and control action
        Return: air penalty w.r.t. CO2(mu) OR CO2(upper bound)
        """
        mu, std = self.get_prediction(g, cnt, f)
        c1 = self.criterion(mu, self.target_co2)
        return c1
    
    def get_energy(self, u_p):
        """
        Input: the past 12 data, and control action
        Return: energy
        """
        airflow = self.up_normalizer.transform(u_p, inverse=True)[:, [1,3,5,7,9,11]]
        energy = torch.sum(airflow - torch.zeros_like(airflow))
        return energy
    
    def solver_grad(self, g, u_p, f, w1, w2, w3, epochs=100):
        self.Loss = []
        g = g.to(self.device)
        x = torch.tensor(u_p.detach().cpu().numpy()).to(self.device)
        x.requires_grad_(True)
        u_p = u_p.to(self.device)
        mask = torch.zeros_like(x, dtype = torch.bool).to(self.device)
        mask[:, 1:] = True
        optimizer = optim.Adam([x], lr=5e-2)
        #default = 90. * torch.ones(1, 13)
        #default = self.up_normalizer.transform(default.to(device), inverse=False)
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            c1 = self.get_air_penalty(g, x, f)
            c2 = self.get_energy(x)
            c3 = self.criterion(x[:,2::2], u_p[:,2::2])
            loss = w1 * c1 + w2 * c2 + w3 * c3
            loss.backward()
            with torch.no_grad():
                x.grad *= mask
            optimizer.step()
            with torch.no_grad():
                x[:,1::2] = torch.clamp(x[:,1::2], min = self.xmin[:,1::2], max= self.xmax[:,1::2])
                x[:,2::2] = torch.clamp(x[:,2::2], min = self.xmin2[:,2::2], max= self.xmax2[:,2::2])
            self.Loss += [[w1*c1.item(), w2*c2.item(), w3*c3.item(), loss.item()]]
        self.Loss = np.array(self.Loss)
        return x

    def get_raw(self, x, record=None):
        action = self.up_normalizer.transform(x).detach().cpu().numpy()
        print("action is", action)
        return action
    
    def visualize_control(self, x):
        # action (13,)
        action = self.get_raw(x)
        action = np.squeeze(action)
        vents = ['Vent 1', 'Vent 2', 'Vent 3', 'Vent 4', 'Vent 5', 'Vent 6']
        angles = action[2::2]
        flows = action[1::2] / 3.24 * 100
        x = np.arange(len(vents))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 3))
        rects1 = ax.bar(x - width/2, angles, width, label='Angle (deg)')
        rects2 = ax.bar(x + width/2, flows, width, label='Flow Rate')
        ax.set_ylabel('Values')
        ax.set_title('Vent Angles and Flow Rates')
        ax.set_xticks(x)
        ax.set_xticklabels(vents)
        ax.legend()
        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom')
        def autolabelflow(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height/100*3.24:.2f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom')
        autolabel(rects1)
        autolabelflow(rects2)
        plt.tight_layout()
        plt.show()
        print(action)