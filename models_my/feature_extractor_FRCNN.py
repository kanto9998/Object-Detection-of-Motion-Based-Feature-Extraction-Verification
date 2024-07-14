import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

#FRCNN 체크용. 검증에서는안쓰임


# feature map을 추출하는 함수
def extract_features(i_frame_dir):

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # 이미지를 Tensor로 변환
    feature_maps = {}
    for img_name in os.listdir(i_frame_dir):
        img_path = os.path.join(i_frame_dir, img_name)
        img = Image.open(img_path)
        img_tensor = F.to_tensor(img).unsqueeze(0)
    
    print(img_tensor)
    print('input(transformed) shape',img_tensor.shape)
    
    # 모델의 백본을 통해 feature map 추출
    with torch.no_grad():
        feature_map = model.backbone(img_tensor)
    
    feature_maps[img_name] = feature_map
    print(feature_map)
    print('feature_map shape',feature_map.shape)

    # FPN의 최종 feature map (Multi-scale feature maps)
    return feature_maps

