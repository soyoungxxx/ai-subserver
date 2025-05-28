import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# from utils import colorize_mask
from torchvision import transforms
from PIL import Image

from torchvision.models import resnet101
from torchvision.models._utils import IntermediateLayerGetter

BatchNorm2d = nn.BatchNorm2d

PADDING_SIZE = 768
RESTORE_FROM = './weights/aqua/model.pth'
NUM_CLASSES = 2
WATER_CLASSES = [1]

# def detect_aqua(input_path, output_path):
#     model = load_model(NUM_CLASSES, RESTORE_FROM)
#     model.eval()
#     model.cuda()

#     cudnn.enabled = True
#     cudnn.benchmark = True

#     image = cv2.imread(input_path)
#     orig_h, orig_w = image.shape[:2]

#     # 전처리
#     resized = cv2.resize(image, (PADDING_SIZE, PADDING_SIZE))
#     img_tensor = transforms.ToTensor()(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))).unsqueeze(0).cuda()

#     with torch.no_grad():
#         pred = model(img_tensor)
#     pred = F.interpolate(pred, size=(PADDING_SIZE, PADDING_SIZE), mode='bilinear', align_corners=True)
#     pred = pred.cpu().data[0].numpy().transpose(1, 2, 0)
#     pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8)

#     # 원본 해상도로 resize
#     pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

#     # 빨간색 마스크 생성
#     color_mask = np.zeros_like(image)
#     for cid in WATER_CLASSES:
#         color_mask[pred == cid] = [0, 0, 255]

#     # alpha blending
#     blended = cv2.addWeighted(image, 1.0, color_mask, 0.9, 0)

#     # 저장
#     cv2.imwrite(output_path, blended)

def detect_aqua(input_path, output_path):
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
            blended = process_frame(model, frame)
            out.write(blended)

        cap.release()
        out.release()
    else:
        image = cv2.imread(input_path)
        blended = process_frame(model, image)
        cv2.imwrite(output_path, blended)

def process_frame(model, frame):
    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (PADDING_SIZE, PADDING_SIZE))
    img_tensor = transforms.ToTensor()(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))).unsqueeze(0).cuda()

    with torch.no_grad():
        pred = model(img_tensor)
    pred = F.interpolate(pred, size=(PADDING_SIZE, PADDING_SIZE), mode='bilinear', align_corners=True)
    pred = pred.cpu().data[0].numpy().transpose(1, 2, 0)
    pred = np.argmax(pred, axis=2).astype(np.uint8)
    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    color_mask = np.zeros_like(frame)
    for cid in WATER_CLASSES:
        color_mask[pred == cid] = [0, 0, 255]
    blended = cv2.addWeighted(frame, 1.0, color_mask, 0.9, 0)
    return blended


def build_model(NUM_CLASSES, RESTORE_FROM):
    model = Aquanet(num_classes=NUM_CLASSES)

    saved_state_dict = torch.load(RESTORE_FROM)
    new_params = model.backbone.state_dict().copy()
    for key, value in saved_state_dict.items():
        if key.split(".")[0] not in ["fc"]:
            new_params[key] = value
    model.backbone.load_state_dict(new_params)

    return model


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

        # 2. Context 모듈 (기존 AquaNet 구조 일부 유지)
        self.context = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
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