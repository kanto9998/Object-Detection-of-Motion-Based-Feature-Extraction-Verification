import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import models, transforms
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt




def extract_features(frame_dir, output_dir, motion_output_dir):
    # F 마지막 레이어 이전까지를 feature extractor로 
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    backbone = model.backbone
    backbone.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(motion_output_dir):
        os.makedirs(motion_output_dir)

    feature_maps = {}
    i_frame_indices = sorted([int(f.split('_')[2].split('.')[0]) for f in os.listdir(frame_dir) if f.startswith('i_frame_')])

    for i_frame_index in i_frame_indices:
        # I-Frame feature map 추출
        i_frame_name = f'i_frame_{i_frame_index}.jpg'
        i_frame_path = os.path.join(frame_dir, i_frame_name)
        img = Image.open(i_frame_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        print('input(transformed) shape', img_tensor.shape)
        
        with torch.no_grad():
            feature_map_dict = backbone(img_tensor)
        feature_maps[i_frame_name] = feature_map_dict

        # 각 스케일의 피쳐맵을 이미지로 저장
        for scale, feature_map in feature_map_dict.items():
            save_feature_map_as_image(feature_map, os.path.join(output_dir, f'feature_map_{i_frame_name}_scale_{scale}.png'))

        # 다음 I-Frame과의 Optical Flow 계산 및 Warping (5월v)
        if i_frame_index + 1 in i_frame_indices:
            next_i_frame_name = f'i_frame_{i_frame_index + 1}.jpg'
            next_i_frame_path = os.path.join(frame_dir, next_i_frame_name)
            img_next = Image.open(next_i_frame_path).convert("RGB")
            motion = estimate_motion(cv2.imread(i_frame_path), cv2.imread(next_i_frame_path))
            print(f'{i_frame_index} warped')
            
            #  생성 및 저장
            for t in range(1, 5):
                warped_feature_map = warp_feature_map(feature_map_dict, motion, t / 4.0)
                for scale, feature_map in warped_feature_map.items():
                    print(f'{i_frame_index} FM saved')
                    save_feature_map_as_image(feature_map, os.path.join(motion_output_dir, f'warped_feature_map_{i_frame_name}_to_{next_i_frame_name}_t{t}_scale_{scale}.png'))

    return feature_maps

def estimate_motion(i_frame, next_i_frame):
    # OpenCV를 사용하여 motion estimation 수행
    motion = cv2.calcOpticalFlowFarneback(cv2.cvtColor(i_frame, cv2.COLOR_BGR2GRAY),
                                          cv2.cvtColor(next_i_frame, cv2.COLOR_BGR2GRAY), 
                                          None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return motion

def warp_feature_map(feature_map_dict, motion, t): #검증용 모션e 사용한 사이즈 와핑 (5월v). 수동메소드는 검증용에는 빼놓음
    warped_feature_map_dict = {}
    for scale, feature_map in feature_map_dict.items():
        
        H, W = feature_map.shape[2], feature_map.shape[3]
        resized_motion = cv2.resize(motion, (W, H))
        warped_feature_map = warp_with_motion(feature_map, resized_motion, t)
        warped_feature_map_dict[scale] = warped_feature_map
    return warped_feature_map_dict

def warp_with_motion(feature_map, motion, t):
    # 시간 t 랑 estimation용 모션 와핑하는 검증메소드
    B, C, H, W = feature_map.size()
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)
    
    flow = t * motion
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    map_x = (grid_x + flow_x).astype(np.float32) / (W - 1) * 2 - 1
    map_y = (grid_y + flow_y).astype(np.float32) / (H - 1) * 2 - 1

    map_x = torch.from_numpy(map_x).unsqueeze(0).unsqueeze(0).to(feature_map.device)
    map_y = torch.from_numpy(map_y).unsqueeze(0).unsqueeze(0).to(feature_map.device)
    grid = torch.stack((map_x, map_y), dim=-1).squeeze(1)

    grid = grid.repeat(B, 1, 1, 1)
    warped_feature_map = torch.nn.functional.grid_sample(feature_map, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped_feature_map

def save_feature_map_as_image(feature_map, output_path):
    feature_map = feature_map.squeeze(0)
    feature_map = torch.sum(feature_map, 0)
    feature_map = feature_map / feature_map.shape[0]

    plt.figure(figsize=(10, 10))
    plt.imshow(feature_map.data.cpu().numpy(), cmap='viridis')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()





##기타
# def extract_features(i_frame_dir):
#     # ResNet50 모델을 불러와 마지막 레이어 이전까지를 feature extractor로 사용
#     model = fasterrcnn_resnet50_fpn(pretrained=True)
#     backbone = model.backbone
#     backbone.eval()
    
#     transform = transforms.Compose([
#         #transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])


#     feature_maps = {}
#     for img_name in os.listdir(i_frame_dir):
#         img_path = os.path.join(i_frame_dir, img_name)
#         img = Image.open(img_path).convert("RGB")
#         img_tensor = transform(img).unsqueeze(0)

#         #print(img_tensor)
#         print('input(transformed) shape',img_tensor.shape)
        
#         with torch.no_grad():
#             feature_map = backbone(img_tensor)
#         feature_maps[img_name] = feature_map
#         #print(img_name)
        
#         #print('feature_map shape',feature_map.shape)
#         # for level_name, feature_map in features.items():
#         #     print(f'feature_map {level_name} shape', feature_map.shape)

# #뒷부분 불필요해서 제거


#     return feature_maps

# # 테스트용으로 한 이미지만
# if __name__ == "__main__":
#     i_frame_dir = 'path_to_i_frame_directory'
#     features = extract_features(i_frame_dir)
#     for key, value in features.items():
#         print(f'{key}: {value.shape}')
#         break




# 레즈넷50 가져오면 마지막 FC레이어까지 통과해버려서 1x1000됨
# def extract_features(i_frame_dir):
#     model = models.resnet50(pretrained=True)
#     model.eval()
    
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     feature_maps = {}
#     for img_name in os.listdir(i_frame_dir):
#         img_path = os.path.join(i_frame_dir, img_name)
#         img = Image.open(img_path)
#         img_tensor = transform(img).unsqueeze(0)
        
#         with torch.no_grad():
#             feature_map = model(img_tensor)
#         feature_maps[img_name] = feature_map

#     #print(feature_maps['i_frame_0045.png'].shape)
#     return feature_maps