import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image
import os
import cv2



def detect_objects(frame_dir, frame_features, output_dir):
    # FPN 체크위해 Faster R-CNN 백본사용
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    i_frame_indices = sorted([int(f.split('_')[2].split('.')[0]) for f in os.listdir(frame_dir) if f.startswith('i_frame_')])

    for i_frame_index in i_frame_indices:
        # I-Frame 객체 검출
        i_frame_name = f'i_frame_{i_frame_index}.jpg'
        i_frame_path = os.path.join(frame_dir, i_frame_name)
        img = Image.open(i_frame_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # feature map 가져오기
        feature_map = frame_features[i_frame_name]

        # Faster R-CNN 모델을 사용하여 객체 검출 수행
        with torch.no_grad():
            prediction = model(img_tensor)

        # 결과를 이미지로 시각화 및 저장
        image = cv2.imread(i_frame_path)
        for element in prediction[0]['boxes']:
            box = element.int().tolist()
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # 결과 저장
        output_path = os.path.join(output_dir, i_frame_name)
        cv2.imwrite(output_path, image)
        print(f'{i_frame_name} : detected')

        # Warping된 feature maps 객체 검출
        for t in range(1, 5):
            for scale in ['scale_0', 'scale_1', 'scale_2', 'scale_3']:
                warped_feature_map_name = f'warped_{i_frame_name}_to_i_frame_{i_frame_index + 1}.jpg_t{t}_scale_{scale}.png'
                if warped_feature_map_name in frame_features:
                    warped_feature_map = frame_features[warped_feature_map_name]
                    with torch.no_grad():
                        prediction = model(img_tensor)

                    # 결과를 이미지로 시각화 및 저장(확인용)
                    image = cv2.imread(i_frame_path)
                    for element in prediction[0]['boxes']:
                        box = element.int().tolist()
                        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        print(f'{i_frame_name}, {warped_feature_map_name} : predicted')

                    # 결과 저장(검증확인용으로만 저장)
                    output_path = os.path.join(output_dir, warped_feature_map_name)
                    cv2.imwrite(output_path, image)
                    print(f'{i_frame_name}, {warped_feature_map_name} : detected')



#  i, p프레임 전부 탐지하는 테스트용

# def detect_objects(frame_dir, frame_features, output_dir):
#     # FPN
#     model = fasterrcnn_resnet50_fpn(pretrained=True)
#     model.eval()

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     i_frame_indices = sorted([int(f.split('_')[2].split('.')[0]) for f in os.listdir(frame_dir) if f.startswith('i_frame_')])

#     for i_frame_index in i_frame_indices:
#         # I-Frame 객체 검출
#         i_frame_name = f'i_frame_{i_frame_index}.jpg'
#         i_frame_path = os.path.join(frame_dir, i_frame_name)
#         img = Image.open(i_frame_path).convert("RGB")
#         img_tensor = transform(img).unsqueeze(0)

#         # feature map 가져오기
#         feature_map = frame_features[i_frame_name]

#         # Faster R-CNN 모델을 사용하여 객체 검출 수행
#         with torch.no_grad():
#             prediction = model(img_tensor)

#         # 결과를 이미지로 시각화 및 저장
#         image = cv2.imread(i_frame_path)
#         for element in prediction[0]['boxes']:
#             box = element.int().tolist()
#             cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

#         # 결과 저장
#         output_path = os.path.join(output_dir, i_frame_name)
#         cv2.imwrite(output_path, image)

#         # Corresponding P-Frames 객체 검출
#         p_frame_files = sorted([f for f in os.listdir(frame_dir) if f.startswith(f'p_frame_{i_frame_index}_')])
#         for p_frame_name in p_frame_files:
#             p_frame_path = os.path.join(frame_dir, p_frame_name)
#             img = Image.open(p_frame_path).convert("RGB")
#             img_tensor = transform(img).unsqueeze(0)

#             # feature map 가져오기
#             feature_map = frame_features[p_frame_name]

#             # Faster R-CNN 모델을 사용하여 객체 검출 수행
#             with torch.no_grad():
#                 prediction = model(img_tensor)

#             # 결과를 이미지로 시각화 및 저장
#             image = cv2.imread(p_frame_path)
#             print(f'i{i_frame_name}, p{p_frame_name} : detected')
#             for element in prediction[0]['boxes']:
#                 box = element.int().tolist()
#                 cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

#             # 결과 저장
#             output_path = os.path.join(output_dir, p_frame_name)
#             cv2.imwrite(output_path, image)

##  RCFN테스트 확인용. 퍼포먼스 테스트용
# import torch
# #from torchvision.models.detection import yolov5
# import torch.nn as nn
# import torchvision.models as models
# #from torchvision.models.detection import r_fcn_resnet50


# from torchvision.models.detection import fasterrcnn_resnet50_fpn

# # Faster R-CNN 모델을 전역 변수로 선언하여 한 번만 로드되도록 설정
# faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)

# class CustomFasterRCNN(nn.Module):
#     def __init__(self, faster_rcnn_model):
#         super(CustomFasterRCNN, self).__init__()
#         self.faster_rcnn_model = faster_rcnn_model

#     def forward(self, feature_map, image_shapes):
#         # RPN을 사용하여 proposals 생성

        
#         proposals, _ = self.faster_rcnn_model.rpn(feature_map, image_shapes)
        
#         # ROI Pooling 및 Box Head를 사용하여 객체 탐지
#         box_features = self.faster_rcnn_model.roi_heads.box_roi_pool(feature_map, proposals, image_shapes)
#         box_features = self.faster_rcnn_model.roi_heads.box_head(box_features)
#         class_logits, box_regression = self.faster_rcnn_model.roi_heads.box_predictor(box_features)
        
#         return proposals, class_logits, box_regression

# # CustomFasterRCNN 모델 인스턴스 생성
# custom_faster_rcnn = CustomFasterRCNN(faster_rcnn_model)

# def detect_objects(i_frame_features):
#     #  [224, 224]
#     image_shapes = [(224,224)]
#     print('fm type',type(i_frame_features),'imageshapes type',type(image_shapes))
#     # 객체 탐지 수행
#     with torch.no_grad():
#         proposals, class_logits, box_regression = custom_faster_rcnn(i_frame_features, image_shapes)
    
#     return proposals, class_logits, box_regression

# r-fcn이 토치비전에 없다네? 
#  한 번만 로드되도록 설정
# #rfcn_model = r_fcn_resnet50(pretrained=True)
# rfcn_model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)


# class CustomRFCN(nn.Module):
#     def __init__(self, rfcn_model):
#         super(CustomRFCN, self).__init__()
#         self.rfcn_model = rfcn_model

#     def forward(self, feature_map, original_image_sizes):
#         #모델 입력
#         proposals, _ = self.rfcn_model.rpn(feature_map, original_image_sizes)
#         box_features = self.rfcn_model.roi_heads.box_roi_pool(feature_map, proposals, original_image_sizes)
#         box_features = self.rfcn_model.roi_heads.box_head(box_features)
#         class_logits, box_regression = self.rfcn_model.roi_heads.box_predictor(box_features)
        
#         return proposals, class_logits, box_regression
    
#resnet에서나온 [1,2048,7,7]

