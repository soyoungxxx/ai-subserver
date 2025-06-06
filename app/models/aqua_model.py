import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from torchvision.models import resnet101
from torchvision.models._utils import IntermediateLayerGetter

BatchNorm2d = nn.BatchNorm2d

PADDING_SIZE = 768
RESTORE_FROM = './weights/aqua/model003.pth'
NUM_CLASSES = 2
WATER_CLASSES = [1]

def detect_aqua(input_path, output_path, polygon_data=None):
    model = load_model(NUM_CLASSES, RESTORE_FROM)
    model.eval().cuda()

    ext = os.path.splitext(input_path)[-1].lower()

    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
    if is_video:
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            blended = process_frame(model, frame, polygon_data)
            out.write(blended)

        cap.release()
        out.release()
    else:
        image = cv2.imread(input_path)
        blended = process_frame(model, image, polygon_data)
        cv2.imwrite(output_path, blended)

def process_frame(model, frame, polygon_data=None):
    orig_h, orig_w = frame.shape[:2]

    # 전체 이미지로 탐지
    resized = cv2.resize(frame, (PADDING_SIZE, PADDING_SIZE))
    img_tensor = transforms.ToTensor()(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))).unsqueeze(0).cuda()

    with torch.no_grad():
        pred = model(img_tensor)

    pred = F.interpolate(pred, size=(PADDING_SIZE, PADDING_SIZE), mode='bilinear', align_corners=True)
    pred = pred.cpu().data[0].numpy().transpose(1, 2, 0)
    pred = np.argmax(pred, axis=2).astype(np.uint8)
    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # ROI 마스크 생성
    roi_mask = None
    if polygon_data:
        roi_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for poly in polygon_data:
            pts = np.array([[int(p.x), int(p.y)] for p in poly.points], dtype=np.int32)
            cv2.fillPoly(roi_mask, [pts], 255)

    # 결과 마스크 생성
    color_mask = np.zeros_like(frame)
    for cid in WATER_CLASSES:
        mask = (pred == cid)
        if roi_mask is not None:
            mask = np.logical_and(mask, roi_mask == 255)
        color_mask[mask] = [0, 0, 255]

    blended = cv2.addWeighted(frame, 1.0, color_mask, 0.9, 0)

    # ROI 윤곽선 시각화
    if polygon_data:
        for poly in polygon_data:
            pts = np.array([[int(p.x), int(p.y)] for p in poly.points], dtype=np.int32)
            cv2.polylines(blended, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    return blended

def load_model(NUM_CLASSES, RESTORE_FROM):
    model = Aquanet(num_classes=NUM_CLASSES)

    saved_state_dict = torch.load(RESTORE_FROM)
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    return model

class Aquanet(nn.Module):
    def __init__(self, num_classes=2):
        super(Aquanet, self).__init__()

        # 1. ResNet101 백본 (pretrained)
        resnet = resnet101(weights="DEFAULT")
        return_layers = {
            "layer4": "out",  # 마지막 feature map만 사용
        }
        self.backbone = IntermediateLayerGetter(resnet, return_layers=return_layers)

        # 2. Context 모듈
        self.context = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        # 3. 최종 분류 레이어
        self.cls = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        x = features["out"]  # ResNet layer4 출력: (B, 2048, H/32, W/32)

        x = self.context(x)  # (B, 256, H/32, W/32)
        out = self.cls(x)    # (B, 2, H/32, W/32)

        return out  # pred_aux 없이 dummy 반환 (기존 구조 호환용)