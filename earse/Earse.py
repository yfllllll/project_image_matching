from mmdet.apis import DetInferencer
from mobile_sam import sam_model_registry, SamPredictor
import numpy as np
import torch
import cv2

class Mask:
    def __init__(self, device=3):
        # 加载模型
        sam_checkpoint = "/data/code/120/weights/mobilesam/weight/mobile_sam.pt"
        #"/data1/yfl/weights/sam_hp/sam_hq_vit_h.pth" #
        model_type = "vit_t"
        #"vit_h" #
        self.device = device

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam.eval()
        self.predictor = SamPredictor(sam)

        self.mmgroundino_inferencer = DetInferencer(
            model='/data/code/120/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py',
            weights="/data/code/120/weights/groundingdino/grounding_dino_swin-l_pretrain_all-56d69e78.pth",
            device=device
        )  # MM-GroundingDINO 模型
    
    def get_bbox(self, mmgroundino_results,conf):
        scores = np.array(mmgroundino_results['predictions'][0]['scores'])
        index = scores > conf
        bbox = np.array(mmgroundino_results['predictions'][0]['bboxes'])
        det_boxes = bbox[index]
        
        return det_boxes        

    def dilate_mask(self, mask, dilation_iterations=2):
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
        return dilated_mask   

    def get_mask(self, image, target_classes='building, tree, mountain, pole, sky', conf=0.15):
        """
        image = cv2.imread(image_path) in BGR order 
        
        """
        height, width, _ = image.shape
        mask = np.zeros(shape=(height, width), dtype=np.uint8)
        target_classes_list = [cls.strip() for cls in target_classes.split(",")]
        text_prompt = " . ".join(target_classes_list) + " ."
        mmgroundino_results = self.mmgroundino_inferencer(
        inputs=image, texts=text_prompt, pred_score_thr=conf, return_vis=True
        )
        transformed_boxes = self.get_bbox(mmgroundino_results, conf)
        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
        transformed_boxes = torch.from_numpy(transformed_boxes).to(self.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(transformed_boxes, image.shape[:2])
        masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
        masks = masks.squeeze(1).cpu().numpy()
        for mask_i in masks:
            mask = np.maximum(mask, mask_i)
        
        mask = self.dilate_mask(mask, 2)
        mask = np.clip(mask, 0, 255)
        inverted_mask = cv2.bitwise_not(mask*255)
        return inverted_mask