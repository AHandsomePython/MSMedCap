import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from PIL import Image
import numpy as np
from lavis.models import load_model_and_preprocess

from torch.utils.data import Dataset, DataLoader

from transformers import SamModel, SamProcessor
from torchvision.transforms import ToPILImage
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
from tqdm import tqdm


device = torch.device("cuda:2")
# image_dir = "/data/xcg/lavis_data/coco-2023us/images/train2014"
# csv_file = "/data/xcg/lavis_data/coco-2023us/excels/translated.csv"
# output_dir = "/data2/xcg_data/lavis_data/2023us/features/clip_features"

image_dir = "/data2/xcg_data/lavis_data/Breast_images/test" # train or test
csv_file = "/data2/xcg_data/lavis_data/Breast_images/labels.csv"
output_dir = "/data2/xcg_data/lavis_data/Breast_images/features/clip_features"
batch_size = 1


model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device)
visual_encoder = model.visual_encoder
ln_vision = model.ln_vision

def clip_transform(raw_image, device=device):
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    return image[0]


class DynamicImageDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.data_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        # self.image_files = []
        # with open(csv_file, 'r') as file:
        #     reader = csv.reader(file)
        #     for row in reader:
        #         self.image_files.append(row[0])   

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_filename = self.image_files[index]
        img_path = os.path.join(self.data_dir, img_filename)
        
        # 使用PIL库加载图片
        img = Image.open(img_path)
        img = img.convert('L')
        img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # 使用图片文件名作为标签
        label = img_filename
        
        return img, label


custom_dataset = DynamicImageDataset(image_dir=image_dir,
                                csv_file=csv_file,
                                transform=clip_transform)
dataloader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=False)


for images, flow_indexs in tqdm(dataloader):
    clip_image_embeddings = ln_vision(visual_encoder(images)).detach()

    for clip_image_embedding, flow_index in zip(clip_image_embeddings, flow_indexs):
        clip_embedding_shape = clip_image_embedding.shape
        clip_feature = clip_image_embedding.view(clip_embedding_shape[0], -1)
        clip_features_numpy = clip_feature.cpu().numpy()
        np.savez_compressed(os.path.join(output_dir, f"{flow_index}.npz"),
                            arr = clip_features_numpy)

    