import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


from transformers import SamModel, SamProcessor
from torchvision.transforms import ToPILImage
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
from tqdm import tqdm


device = torch.device("cuda:4")
# image_dir = "/data/xcg/lavis_data/coco-2023us/images/train2014"
# csv_file = "/data/xcg/lavis_data/coco-2023us/excels/translated.csv"
# output_dir = "/data2/xcg_data/lavis_data/2023us/features/sam_features"

image_dir = "/data2/xcg_data/lavis_data/Breast_images/test" # train or test
csv_file = "/data2/xcg_data/lavis_data/Breast_images/labels.csv"
output_dir = "/data2/xcg_data/lavis_data/Breast_images/features/sam_features"
batch_size = 4


sam_model = SamModel.from_pretrained("facebook/sam-vit-big").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-big")


def samtransform(raw_image, device=device):
    image = sam_processor(raw_image, return_tensors="pt").to(device)['pixel_values']
    return image


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
        
        if self.transform:
            img = self.transform(img)
        
        # 使用图片文件名作为标签
        label = img_filename
        
        return img[0], label


custom_dataset = DynamicImageDataset(image_dir=image_dir,
                                csv_file=csv_file,
                                transform=samtransform)
dataloader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=False)


for images, flow_indexs in tqdm(dataloader):
    sam_image_embeddings = sam_model.get_image_embeddings(images).detach()

    for sam_image_embedding, flow_index in zip(sam_image_embeddings, flow_indexs):
        sam_embedding_shape = sam_image_embedding.shape
        sam_feature = sam_image_embedding.view(sam_embedding_shape[0], -1)
        sam_features_numpy = sam_feature.cpu().numpy()
        np.savez_compressed(os.path.join(output_dir, f"{flow_index}.npz"),
                            arr = sam_features_numpy)
