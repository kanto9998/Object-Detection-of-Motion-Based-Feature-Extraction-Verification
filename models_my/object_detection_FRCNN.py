import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image


#FRCNN 체크용. 검증에서는안쓰임
# Faster R-CNN 모델 로드
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


def detect_objects(i_frame_features):
    # 이미지를 Tensor로 변환
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Add batch dimension

    # 객체 탐지 수행
    with torch.no_grad():
        detections = model(image_tensor)

    return detections