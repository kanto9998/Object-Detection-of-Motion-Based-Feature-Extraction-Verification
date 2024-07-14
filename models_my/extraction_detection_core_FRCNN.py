import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.resnet import resnet50
from torchvision.transforms import functional as F
from collections import OrderedDict
from PIL import Image
import os

# Faster R-CNN 모델 로드 (백본만 ResNet-50으로 사용) #FRCNN 체크용. 검증에서는안쓰임
backbone = resnet50(pretrained=True)
# Feature Pyramid Network을 사용하지 않음
backbone = torch.nn.Sequential(OrderedDict([
    ('body', torch.nn.Sequential(*list(backbone.children())[:-2]))
]))
backbone.out_channels = 2048

anchor_generator = AnchorGenerator(
    sizes=((32,), (64,), (128,), (256,), (512,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=7,
    sampling_ratio=2
)

model = FasterRCNN(backbone,
                   num_classes=91,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

model.eval()

def detect_objects(feature_map, image_sizes):
    # feature_map은 dict 형태로 제공되어야 함
    # image_sizes는 이미지의 원래 크기 (H, W)

    print('fm type:', type(feature_map), 'image_sizes type:', type(image_sizes))

    with torch.no_grad():
        proposals, _ = model.rpn(feature_map, image_sizes)
        detections, _ = model.roi_heads(feature_map, proposals, image_sizes)

    # 탐지 결과를 더 자세히 출력
    detailed_detections = []
    for i, detection in enumerate(detections):
        boxes = detection['boxes']
        labels = detection['labels']
        scores = detection['scores']
        for j in range(len(boxes)):
            detailed_detections.append({
                'box': boxes[j].cpu().numpy(),
                'label': labels[j].cpu().numpy(),
                'score': scores[j].cpu().numpy()
            })
            print(f"Detection {i+1}-{j+1}: Label: {labels[j].item()}, Score: {scores[j].item()}, Box: {boxes[j].tolist()}")

    return detailed_detections

def extract_features(i_frame_dir):
    feature_maps = {}
    for img_name in os.listdir(i_frame_dir):
        img_path = os.path.join(i_frame_dir, img_name)
        img = Image.open(img_path)
        image_tensor = F.to_tensor(img).unsqueeze(0)

        with torch.no_grad():
            feature_map = model.backbone(image_tensor)

        feature_maps[img_name] = feature_map
        print('Feature map shape:', feature_map.shape)

    return feature_maps



# import torch
# import torchvision
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.models.resnet import resnet50
# from torchvision.transforms import functional as F
# from collections import OrderedDict
# from PIL import Image
# import os
# # Faster R-CNN 모델 로드 (백본만 ResNet-50으로 사용)
# backbone = resnet50(pretrained=True)
# # Feature Pyramid Network을 사용하지 않음
# backbone = torch.nn.Sequential(OrderedDict([
#     ('body', torch.nn.Sequential(*list(backbone.children())[:-2]))
# ]))
# backbone.out_channels = 2048

# anchor_generator = AnchorGenerator(
#     sizes=((32,), (64,), (128,), (256,), (512,)),
#     aspect_ratios=((0.5, 1.0, 2.0),) * 5
# )


# roi_pooler = torchvision.ops.MultiScaleRoIAlign(
#     featmap_names=['0'],
#     output_size=7,
#     sampling_ratio=2
# )

# model = FasterRCNN(backbone,
#                    num_classes=91,
#                    rpn_anchor_generator=anchor_generator,
#                    box_roi_pool=roi_pooler)

# model.eval()

# # # 객체 탐지 함수 (feature map을 입력으로 받음)
# # def detect_objects(feature_map, image_sizes=[(224,224)]):
# #     # features는 dict 형태로 제공되어야 함
# #     # image_sizes는 이미지의 원래 크기 (H, W)

# #     print('fm type',type(feature_map),'imageshapes type',type(image_sizes))

# #     with torch.no_grad():
# #         detections = model.rpn(feature_map, image_sizes)
# #         detections = model.roi_heads(feature_map, detections, image_sizes)
    
# #     return detections

# def detect_objects(feature_map, image_sizes):
#     # features는 dict 형태로 제공되어야 함
#     # image_sizes는 이미지의 원래 크기 (H, W)

#     print('fm type:', type(feature_map), 'image_sizes type:', type(image_sizes))

#     with torch.no_grad():
#         proposals, _ = model.rpn(feature_map, image_sizes)
#         detections, _ = model.roi_heads(feature_map, proposals, image_sizes)

#     # 탐지 결과를 더 자세히 출력
#     detailed_detections = []
#     for i, detection in enumerate(detections):
#         boxes = detection['boxes']
#         labels = detection['labels']
#         scores = detection['scores']
#         for j in range(len(boxes)):
#             detailed_detections.append({
#                 'box': boxes[j].cpu().numpy(),
#                 'label': labels[j].cpu().numpy(),
#                 'score': scores[j].cpu().numpy()
#             })
#             print(f"Detection {i+1}-{j+1}: Label: {labels[j].item()}, Score: {scores[j].item()}, Box: {boxes[j].tolist()}")

#     return detailed_detections

# def extract_features(i_frame_dir):
#     feature_maps = {}
#     for img_name in os.listdir(i_frame_dir):
#         img_path = os.path.join(i_frame_dir, img_name)
#         img = Image.open(img_path)
#         image_tensor = F.to_tensor(img).unsqueeze(0)

#         with torch.no_grad():
#             feature_map = model.backbone(image_tensor)

#         feature_maps[img_name] = feature_map
#         print(feature_map)
#         print('feature_map shape',feature_map.shape)

    

#     return feature_maps #임시