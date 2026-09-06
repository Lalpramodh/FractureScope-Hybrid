# models/fracture_model.py
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.models import swin_t, Swin_T_Weights

class FractureModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FractureModel, self).__init__()
        
        # Load pretrained YOLOv8
        self.yolo = YOLO("yolov8n.pt")  # use smaller variant for speed; can be yolov8s/m/l/x
        self.yolo.eval()
        
        # Swin Transformer backbone
        self.swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.swin.head = nn.Identity()  # remove final classifier
        
        # Final classifier
        self.fc = nn.Linear(768, num_classes)  # swin_t output is 768-dim

    def forward(self, x):
        with torch.no_grad():
            # Get bounding boxes from YOLO
            results = self.yolo(x)
            boxes = results[0].boxes.xyxy  # assuming one image at a time
            if len(boxes) == 0:
                return torch.tensor([[0.5, 0.5]])  # uncertain if no detection

            # Crop first detected box (could be improved for batch)
            x1, y1, x2, y2 = map(int, boxes[0])
            x = x[..., y1:y2, x1:x2]  # crop

        # Pass cropped patch to Swin Transformer
        x = self.swin(x)
        out = self.fc(x)
        return out
