import torch
from models.SamBlip import SamBlip
import numpy as np

model = SamBlip()

device = torch.device("cuda:0")

model = model.to(device)

clip_features = np.load("/data2/xcg_data/lavis_data/2023us/features/clip_features/20230101_11548959_012117568.npz")['arr']
sam_features = np.load("/data2/xcg_data/lavis_data/2023us/features/sam_features/20230101_11548959_012117568.npz")['arr']
clip_features = torch.tensor(clip_features).to(device).unsqueeze(0)
sam_features = torch.tensor(sam_features).to(device).unsqueeze(0)

output = model.generate(clip_features=clip_features, sam_features=sam_features, prompt="")
print(output)