import numpy as np
import cv2


## 3월 체크용버전임 feature_extractor로
def estimate_motion(i_frame, p_frame):
    # OpenCV를 사용하여 motion estimation 수행
    motion = cv2.calcOpticalFlowFarneback(cv2.cvtColor(i_frame, cv2.COLOR_BGR2GRAY),
                                          cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY), 
                                          None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return motion

def warp_features(p_frame_dir, i_frame_features):
    warped_features = {}
    for p_frame_name in os.listdir(p_frame_dir):
        p_frame_path = os.path.join(p_frame_dir, p_frame_name)
        p_frame = cv2.imread(p_frame_path)
        
        i_frame_name = find_corresponding_i_frame(p_frame_name)
        i_frame = cv2.imread(os.path.join('data/i_frames/', i_frame_name))
        
        motion = estimate_motion(i_frame, p_frame)
        
        # Feature map warping 수행
        feature_map = i_frame_features[i_frame_name]
        warped_feature_map = warp_feature_map(feature_map, motion)
        warped_features[p_frame_name] = warped_feature_map
    return warped_features

def warp_feature_map(feature_map, motion):

    h, w = motion.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x = (grid_x + motion[..., 0]).astype(np.float32)
    grid_y = (grid_y + motion[..., 1]).astype(np.float32)
    warped_map = cv2.remap(feature_map, grid_x, grid_y, interpolation=cv2.INTER_LINEAR)
    return warped_map
