import os
import timm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_MIRROR"] = 'https://hf-mirror.com'
import torch

import numpy as np
import torch.nn as nn
from PIL import Image
from urllib.request import urlopen
from thop import profile
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from tqdm import tqdm
import torch.nn.functional as F

weight_locs: dict[str, str] = {}
weight_locs["cross_area"] = "/data/code/120/image-matching-models/uavloc_weights/vit_base_eva_gta_cross_area.pth"
weight_locs["same_area"] = "/data/code/120/image-matching-models/uavloc_weights/vit_base_eva_gta_same_area.pth"
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

def is_image(file_path):
    """Check if the file is a valid image by verifying its extension and attempting to open it."""
    # Check if the file has a valid image extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in VALID_IMAGE_EXTENSIONS:
        return False
    
    # Check if the file can be opened as an image
    return True

class MLP(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, output_size=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class DesModel(nn.Module):

    def __init__(self, 
                 checkpoint = "cross_area",
                 model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k',
                 pretrained=True,
                 img_size=384,
                 share_weights=True,
                 train_with_recon=False,
                 train_with_offset=False,
                 model_hub='timm'):
                 
        super(DesModel, self).__init__()
        self.share_weights = share_weights
        self.model_name = model_name
        self.img_size = img_size
        if share_weights:
            if "vit" in model_name or "swin" in model_name:
                # automatically change interpolate pos-encoding to img_size
                self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
            else:
                self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        else:
            if "vit" in model_name or "swin" in model_name:
                self.model1 = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
                self.model2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
            else:
                self.model1 = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
                self.model2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        if train_with_offset:
            self.MLP = MLP()
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if checkpoint is not None:
            if checkpoint not in weight_locs.keys():
                raise ValueError(f"pretrained should be None or one of {weight_locs.keys()}")

            pretrained_dict = torch.load(weight_locs[checkpoint], map_location=torch.device("cpu"))

            # 获取模型的状态字典
            model_dict = self.state_dict()

            # 过滤掉不匹配的键
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 更新模型的状态字典
            model_dict.update(pretrained_dict)

            # 加载更新后的状态字典
            self.load_state_dict(model_dict)
            # self.load_state_dict(pretrained_dict["state_dict"])
        self.eval()

    def get_config(self,):
        if self.share_weights:
            data_config = timm.data.resolve_model_data_config(self.model)
        else:
            data_config = timm.data.resolve_model_data_config(self.model1)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        if self.share_weights:
            self.model.set_grad_checkpointing(enable)
        else:
            self.model1.set_grad_checkpointing(enable)
            self.model2.set_grad_checkpointing(enable)

    def freeze_layers(self, frozen_blocks=10, frozen_stages=[0,0,0,0]):
        pass

    def forward(self, img1=None, img2=None):

        if self.share_weights:
            if img1 is not None and img2 is not None:
                image_features1 = self.model(img1)     
                image_features2 = self.model(img2)
                return image_features1, image_features2            
            elif img1 is not None:
                image_features = self.model(img1)
                return image_features
            else:
                image_features = self.model(img2)
                return image_features
        else:
            if img1 is not None and img2 is not None:
                image_features1 = self.model1(img1)     
                image_features2 = self.model2(img2)
                return image_features1, image_features2            
            elif img1 is not None:
                image_features = self.model1(img1)
                return image_features
            else:
                image_features = self.model2(img2)
                return image_features

    def offset_pred(self, img_feature1, img_feature2):
        offset = self.MLP(torch.cat((img_feature1, img_feature2), dim=1))
        return offset


class Sample4Geo(nn.Module):
    def __init__(self, device="cpu", *args, **kwargs):
        super(Sample4Geo, self).__init__()
        self.device = device
        self.model = DesModel(checkpoint = "cross_area").to(self.device)
        data_config = self.model.get_config()
        mean = data_config["mean"]
        std = data_config["std"]
        img_size = (self.model.img_size, self.model.img_size)
        self.transforms =  A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])    

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        # 将 BGR 转换为 RGB 格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transforms(image=img)['image'].to(self.device)

    def get_img_feature(self, image):
        with torch.no_grad():
            img_feature = self.model(img1=image)
            return img_feature

    def get_img_features_batch(self, images):
        # 对整个图像批次进行特征提取
        with torch.no_grad():
            img_features = self.model(img2=images)
            return img_features

    def retrieve_topk(self, query_img_folder, gallery_img_folder, top_k=6, gallery_batch_size=64):
        query_img_paths = []
        gallery_img_paths = []

        # 判断 query_img_folder 是否为文件路径或文件夹
        if os.path.isdir(query_img_folder):
            query_img_paths = [os.path.join(query_img_folder, f) for f in os.listdir(query_img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        elif os.path.isfile(query_img_folder) and not is_image(query_img_folder):
            with open(query_img_folder, 'r') as f:
                query_img_paths = [line.strip() for line in f.readlines()]
        elif os.path.isfile(query_img_folder) and is_image(query_img_folder):
            query_img_paths = [query_img_folder]  # 如果是直接传递的文件路径
        else:
            query_img_paths = query_img_folder # 如果是直接传递的列表
        # 判断 gallery_img_folder 是否为文件路径、文件夹或列表
        if isinstance(gallery_img_folder, list):  # 如果传入的是图像路径的列表
            gallery_img_paths = gallery_img_folder
        elif os.path.isdir(gallery_img_folder):  # 如果是文件夹路径
            gallery_img_paths = [os.path.join(gallery_img_folder, f) for f in os.listdir(gallery_img_folder) if f.endswith(VALID_IMAGE_EXTENSIONS)]
        elif os.path.isfile(gallery_img_folder):  # 如果是文件路径
            with open(gallery_img_folder, 'r') as f:
                gallery_img_paths = [line.strip() for line in f.readlines()]

        # 初始化相似度矩阵
        similarities = torch.zeros((len(query_img_paths), len(gallery_img_paths))).to(self.device)

        # 逐个查询图像提取特征并计算与所有检索图像的相似度
        for query_idx, query_img_path in tqdm(enumerate(query_img_paths), desc="Processing Query Images", total=len(query_img_paths)):
            query_image = self.preprocess(query_img_path)
            query_image_batch = query_image.unsqueeze(0)  # 在第0维增加一个维度
            query_feature = self.get_img_feature(query_image_batch)

            # 归一化查询图像特征
            query_feature_normalized = F.normalize(query_feature, dim=-1)

            # 逐批加载检索图像并计算相似度
            for start_idx in tqdm(range(0, len(gallery_img_paths), gallery_batch_size), desc="Processing Gallery Images"):
                end_idx = min(start_idx + gallery_batch_size, len(gallery_img_paths))
                batch_gallery_paths = gallery_img_paths[start_idx:end_idx]

                # 批量加载检索图像
                gallery_images = [self.preprocess(gallery_img_path) for gallery_img_path in batch_gallery_paths]
                gallery_images_tensor = torch.stack(gallery_images)
                batch_gallery_features = self.get_img_features_batch(gallery_images_tensor)

                # 归一化检索图像特征
                batch_gallery_features_normalized = F.normalize(batch_gallery_features, dim=-1)

                # 计算查询图像与当前批次检索图像的相似度
                similarities[query_idx, start_idx:end_idx] = query_feature_normalized @ batch_gallery_features_normalized.T

        # 获取每个查询图像最相似的 top-k 图像
        topk_results = {}
        for query_idx, query_img_path in enumerate(query_img_paths):
            # 获取 top-k 相似图像的索引
            topk_idx = similarities[query_idx].argsort(descending=True)[:top_k]
            topk_paths = [gallery_img_paths[i] for i in topk_idx]
            topk_results[query_img_path] = topk_paths

        return topk_results

    def forward(self, query_img_folder, gallery_img_folder, top_k=5, gallery_batch_size=64):
        return self.retrieve_topk(query_img_folder, gallery_img_folder, top_k=top_k, gallery_batch_size=64)










if __name__ == '__main__':
    # model = TimmModel(model_name='timm/vit_large_patch16_384.augreg_in21k_ft_in1k')
    # # model = TimmModel(model_name='timm/vit_base_patch16_224.augreg_in1k')
    # # from timm.models.vision_transformer import vit_base_patch16_224
    # # model = vit_base_patch16_224(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, num_classes=0)


    # model = DesModel(model_name='timm/resnet101.tv_in1k', img_size=384)
    # model = DesModel(model_name='convnext_base.fb_in22k_ft_in1k_384', img_size=384)
    model = DesModel(model_name='timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k', img_size=384)
    # # model = TimmModel(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k')
    # # model = TimmModel(model_name='timm/vit_medium_patch16_rope_reg1_gap_256.sbb_in1k')
    # # model = TimmModel(model_name='timm/vit_medium_patch16_gap_256.sw_in12k_ft_in1k')
    # # model = TimmModel(model_name='timm/resnet101.tv_in1k') 
    # # img = Image.open(urlopen(
    # # 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    # # ))
    x = torch.rand((1, 3, 384, 384))
    x = x.cuda()
    model.cuda()
    x = model(x)
    print(x.shape)

    # flops, params = profile(model, inputs=(x,))
    # # print(img.size)
    # # img = transform(img)
    # # print(img.size)

    # # print(model1)
    # print('flops(G)', flops/1e9, 'params(M)', params/1e6)

    # from transformers import CLIPProcessor, CLIPModel
    # model = CLIPModel.from_pretrained("/home/xmuairmud/jyx/clip-vit-base-patch16")
    # vision_model = model.vision_model
    # print(vision_model)

    # dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    # print(dinov2_vitb14_reg.set_grad_checkpointing(True))

    # from transformers import ViTModel, ViTImageProcessor, AutoModelForImageClassification, AutoConfig
    # config = AutoConfig.from_pretrained('facebook/dino-vitb16')
    # config.image_size = 384
    # model = ViTModel.from_pretrained('facebook/dino-vitb16', config=config, ignore_mismatched_sizes=True)
    # model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True, img_size=(384, 384))
    # data_config = timm.data.resolve_model_data_config(model)
    # print(data_config)
    # processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')


    # x = torch.rand((1, 3, 384, 384))
    # inputs = processor(images=x, return_tensors="pt")
    # print(inputs['pixel_values'].shape)
    # outputs = model(**inputs)
    # print(outputs.pooler_output.shape)
    # print(model(x).shape)
    # flops, params = profile(dinov2_vitb14_reg, inputs=(x,))
    # print('flops(G)', flops/1e9, 'params(M)', params/1e6)


