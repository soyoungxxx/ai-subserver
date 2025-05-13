import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNetModel:
    def __init__(self, unet_model_path="./models/unet/model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = (1024, 1024)
        
        self.model = UNet(n_channels=3, n_classes=2).to(self.device)
        state_dict = torch.load(unet_model_path, map_location=self.device)
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, input_path, output_path, polygon_data):
         # 입력 파일 확장자 확인
        is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov'))
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        if is_video:
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, round(fps), (width, height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                resized = cv2.resize(frame, self.image_size)

                # 수체 segmentation
                with torch.no_grad():
                    input_tensor = transform(resized).unsqueeze(0).to(self.device)
                    output = self.model(input_tensor)
                    mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                # mask → 파란색 overlay 만들기
                color_mask = np.zeros_like(frame)
                color_mask[mask == 1] = [255, 0, 0]  # 수체는 파란색 (BGR)
                blended = cv2.addWeighted(frame, 0.8, color_mask, 0.2, 0)

                out.write(blended)
            
            cap.release()
            out.release()

        else:
            # 이미지 추론
            img = cv2.imread(input_path)
            h, w = img.shape[:2]
            resized = cv2.resize(img, (1024, 1024))
            input_tensor = transform(resized).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            color_mask = np.zeros_like(img)
            color_mask[mask == 1] = [255, 0, 0]
            blended = cv2.addWeighted(img, 0.8, color_mask, 0.2, 0)
            cv2.imwrite(output_path, blended)