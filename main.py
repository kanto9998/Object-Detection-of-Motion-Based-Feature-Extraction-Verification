import os
from utils.video_processing import extract_frames
from models_my.feature_extractor import extract_features
from models_my.object_detection import detect_objects
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import subprocess
import cv2

# def motion_to_color_with_white_bg(motion, threshold=6.0): #모션 임시확인용이라 필요없습니다
#     # motion 벡터의 크기와 방향 계산
#     magnitude, angle = cv2.cartToPolar(motion[..., 0], motion[..., 1])

#     # 방향을 [0, 180] 범위로 변환하여 색조(hue)로 사용
#     angle = angle * 180 / np.pi / 2

#     # 크기를 [0, 255] 범위로 변환
#     magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

#     # 움직임이 있는 부분에 색상을 적용하고 나머지는 흰색으로 설정
#     hsv = np.zeros((motion.shape[0], motion.shape[1], 3), dtype=np.uint8)
#     hsv[..., 0] = angle
#     hsv[..., 1] = 255
#     hsv[..., 2] = magnitude

#     # HSV를 BGR로 변환
#     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#     # 흰색 배경 생성
#     white_bg = np.ones_like(bgr, dtype=np.uint8) * 255

#     # 크기가 임계값 이상인 부분에만 색상 적용
#     mask = magnitude > threshold
#     white_bg[mask] = bgr[mask]

#     return white_bg


def main(video_path,frame_path): #체크

    #검증용 프레임 
    extract_frames(frame_path,video_path) #frame_path에 i_frame_0.jpg, p_frame_0_0.jpg 

    # 임시 모션체크(라이브러리)
    # motion = estimate_motion(cv2.imread('data/i_frames/i_frame_0001.png' ),p_frame = cv2.imread('data/p_frames/p_frame_0020.png'))
    # print('모션',motion,type(motion),motion.shape)
    # motion_color_with_white_bg = motion_to_color_with_white_bg(motion)
    # cv2.imwrite('motion_color_with_white_bg.png', motion_color_with_white_bg)

    feature_map_output_dir_temp = 'data/feature_maps_temp/'
    motion_output_dir_temp = 'data/motion_feature_maps/'
    # I-frame에서 feature map 추출. 확인용 . 압축코덱추출본은 별도
    frame_features = extract_features(frame_path,feature_map_output_dir_temp,motion_output_dir_temp)

    #모션 검증용 MPI Sintel불러와야함. 본 파일에는 x


    object_detection_results_dir = 'data/object_detection_results/' # 체크용
    os.makedirs(object_detection_results_dir, exist_ok=True)

    detect_objects(frame_path, frame_features, object_detection_results_dir) #사진에서바로 체크. 현재 객체탐지에서 bbox신뢰도랑 객체 json이름 초기화해놓음 무작위로잡힘. 사용시 적절히 수정할것. 

    #RPN수정이전
    #     proposals, class_logits, box_regression = detect_objects(single_item_dict)
    #     print("Proposals:", proposals)
    #     print("Class Logits:", class_logits)
    #     print("Box Regression:", box_regression)


if __name__ == "__main__":
    video_path = "data/raw_videos/example.mp4"
    frame_path ="data/frames/"
    main(video_path,frame_path)




#통합

# i frame 피쳐맵 생성확인까지 완료 feature_map shape torch.Size([1, 2048, 7, 7])
# def main(video_path):
#     # 비디오에서 프레임 추출
#     #frame_dir = 'data/frames/'
#     #extract_frames(video_path, frame_dir)
#     #extract_i, p 함수로 통합시킴 24/05

#     # 프레임 분류 (I-frame, P-frame)
#     i_frame_dir = 'data/i_frames/'
#     p_frame_dir = 'data/p_frames/'
#     #classify_frames(frame_dir, i_frame_dir, p_frame_dir)
#     #extract_i, p 함수로 통합시킴 24/05

#     #나중에 함수 하나로 통합시킬것
        # extract_i_frames(video_path,i_frame_dir)
        # extract_p_frames(video_path,p_frame_dir)


#     # I-frame에서 feature map 추출
#     i_frame_features = extract_features(i_frame_dir)

#     feature_map = i_frame_features['i_frame_0045.png']

#     # P-frame에서 motion과 residual을 사용하여 feature map warping
#     p_frame_features = warp_features(p_frame_dir, i_frame_features)

#     # Object detection 수행
#     detect_objects(i_frame_features, p_frame_features)

# if __name__ == "__main__":
#     video_path = "data/raw_videos/example.mp4"
#     main(video_path)

#코덱미사용 메인
# for i_frame in os.listdir(i_frame_dir):
#     i_frame_path = os.path.join(i_frame_dir, i_frame)

#     image = Image.open(i_frame_path).convert('RGB')
#     # image = transform(image).unsqueeze(0)

#     # feature_map = feature_extractor.extract(image)
#     # print('i_frame : ',i_frame, ',feature map shape',feature_map.shape)
    
#     # P-frame 처리 및 feature map warping
#     for p_frame in os.listdir(p_frame_dir):
#         # p_frame_path = os.path.join(p_frame_dir, p_frame)
#         # warped_feature_map = feature_extractor.warp(p_frame_path, feature_map)

#         # print('i_frame : ',i_frame,',p_frame : ',p_frame, ',warped feature map shape',warped_feature_map.shape,',warped feature map type',type(warped_feature_map))
#         #detections = object_detector.detect(image,warped_feature_map)
#         detections = object_detector.detect(image)
#         print(detections)



## 05vr
# def extract_motion_vectors(video_path):
#     cmd = [
#         'ffmpeg', '-flags2', '+export_mvs', '-i', video_path,
#         '-vf', 'codecview=mv=pf+bf+bb', '-f', 'null', '-'
#     ]
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     return result.stderr

# def parse_motion_vectors(log_data, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     motion_vectors = []
#     lines = log_data.split('\n')
    
#     mv_data = None
#     for line in lines:
#         if 'frame' in line and 'pkt_pts_time' in line:
#             if mv_data:
#                 motion_vectors.append(mv_data)
#             frame_info = re.search(r'frame:\s+(\d+)', line)
#             if frame_info:
#                 frame_num = int(frame_info.group(1))
#                 mv_data = {'frame': frame_num, 'motion_vectors': [], 'residuals': []}
#         elif 'motion vector' in line:
#             mv_info = re.search(r'motion vector\(\s*(-?\d+),\s*(-?\d+)\)\s+(\d+)\s+(\d+)', line)
#             if mv_info:
#                 mv_data['motion_vectors'].append({
#                     'src_x': int(mv_info.group(1)),
#                     'src_y': int(mv_info.group(2)),
#                     'dst_x': int(mv_info.group(3)),
#                     'dst_y': int(mv_info.group(4))
#                 })
#         elif 'residual' in line:
#             residual_info = re.search(r'residual\((.+?)\)', line)
#             if residual_info:
#                 mv_data['residuals'].append(residual_info.group(1))
    
#     if mv_data:
#         motion_vectors.append(mv_data)
    
#     with open(os.path.join(output_dir, 'motion_vectors.json'), 'w') as f:
#         json.dump(motion_vectors, f, indent=4)

# # 이미지 로드 및 전처리 함수
# def load_image(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     image = Image.open(image_path)
#     return transform(image).unsqueeze(0)

# # 특징 추출 함수
# model = models.resnet101(pretrained=True)
# model.eval()

# def extract_features(image_path):
#     image = load_image(image_path)
#     with torch.no_grad():
#         features = model(image)
#     return features

# # LSTM 기반 메모리 네트워크 클래스
# class MotionAidedLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(MotionAidedLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, input_dim)

#     def forward(self, features, motion_vectors, residuals):
#         h, _ = self.lstm(features)
#         h = h + motion_vectors + residuals
#         out = self.fc(h)
#         return out

# memory_network = MotionAidedLSTM(input_dim=2048, hidden_dim=512)

# # 객체 탐지 함수
# def detect_objects(feature_map):
#     model = YOLOv5('yolov5s.pt')
#     results = model(feature_map)
#     return results

# # 비디오 처리 메인 함수
# def process_video(video_path):
#     i_frame_output_dir = 'i_frames'
#     p_frame_output_dir = 'p_frames'
#     motion_data_output_dir = 'motion_data'
    
#     extract_i_frames(video_path, i_frame_output_dir)
#     extract_p_frames(video_path, p_frame_output_dir)
#     log_data = extract_motion_vectors(video_path)
#     parse_motion_vectors(log_data, motion_data_output_dir)

#     i_frame_features = []
#     i_frame_paths = sorted([os.path.join(i_frame_output_dir, f) for f in os.listdir(i_frame_output_dir)])
#     p_frame_paths = sorted([os.path.join(p_frame_output_dir, f) for f in os.listdir(p_frame_output_dir)])

#     for i_frame_path in i_frame_paths:
#         feature = extract_features(i_frame_path)
#         i_frame_features.append(feature)

#     with open(os.path.join(motion_data_output_dir, 'motion_vectors.json'), 'r') as f:
#         motion_data = json.load(f)
    
#     for i, p_frame_path in enumerate(p_frame_paths):
#         print('running')
#         motion_vectors = torch.tensor(motion_data[i]['motion_vectors'])
#         residuals = torch.tensor(motion_data[i]['residuals'])
#         feature = memory_network(i_frame_features[-1], motion_vectors, residuals)
#         detection_results = detect_objects(feature)
#         # 탐지 결과 처리

# if __name__ == "__main__":
#     video_path = 'example.mp4'
#     process_video(video_path)


##도커통합전. 메소드 작성공유
# # 1. 데이터셋. ImageNET 로드
# def extract_frames(video_path):
#     i_frames = []
#     p_frames = []
#     cap = cv2.VideoCapture(video_path)
#     frame_idx = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_type = check_frame_type(frame_idx)
#         if frame_type == 'I':
#             i_frames.append(frame)
#         elif frame_type == 'P':
#             p_frames.append(frame)
        
#         frame_idx += 1
#     cap.release()
#     return i_frames, p_frames

# def check_frame_type(frame_idx):
#     # 적용
#     return 'I' if frame_idx % 10 == 0 else 'P'

# # 2. I
# def process_i_frame(i_frame, model):
#     i_frame_resized = cv2.resize(i_frame, (224, 224)) # Model input size
#     i_frame_normalized = i_frame_resized / 255.0
#     feature_map = model.predict(np.expand_dims(i_frame_normalized, axis=0))
#     return feature_map

# # 3. P
# def process_p_frame(p_frame, motion_vector, residual, i_frame_feature):
#     warped_feature = warp_feature_map(i_frame_feature, motion_vector)
#     corrected_feature = apply_residual(warped_feature, residual)
#     return corrected_feature

# def warp_feature_map(feature_map, motion_vector):
#     # 적용
#     warped_feature = feature_map #
#     return warped_feature

# def apply_residual(feature_map, residual):
   
#     corrected_feature = feature_map + residual 
#     return corrected_feature

# def detect_objects(feature_map, detection_model):
#     detections = detection_model.detect(feature_map)
#     return detections

# # 메인
# def main(video_path, i_frame_model_path, detection_model_path):
#     i_frames, p_frames = extract_frames(video_path)
    
#     i_frame_model = load_model(i_frame_model_path)
#     detection_model = YOLOv5(detection_model_path)
    
#     for i_frame in i_frames:
#         i_frame_feature = process_i_frame(i_frame, i_frame_model)
        
#         for p_frame in p_frames:
#             motion_vector, residual = extract_motion_and_residual(p_frame)
#             p_frame_feature = process_p_frame(p_frame, motion_vector, residual, i_frame_feature)
#             detections = detect_objects(p_frame_feature, detection_model)
#             print(detections)

# def extract_motion_and_residual(p_frame):
#     #  ffmpeg
#     motion_vector = np.zeros_like(p_frame) 
#     residual = np.zeros_like(p_frame) 
#     return motion_vector, residual

# if __name__ == "__main__":
#     video_path = 'path_to_video.mp4'
#     i_frame_model_path = 'path_to_i_frame_model.h5'
#     detection_model_path = 'path_to_detection_model.pt'
#     main(video_path, i_frame_model_path, detection_model_path)
