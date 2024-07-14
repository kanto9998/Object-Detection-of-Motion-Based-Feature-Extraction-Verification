import os
import ffmpeg
import subprocess
import cv2

def extract_frames(frame_path, filename):
    if not os.path.exists(frame_path):
        os.mkdir(frame_path)
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [filename]).decode()
    f_types = out.replace('pict_type=', '').split()
    frame_types = list(zip(range(len(f_types)), f_types))

    i_frames = [x[0] for x in frame_types if x[1] == 'I']
    p_frames = [x[0] for x in frame_types if x[1] == 'P']

    if i_frames:
        cap = cv2.VideoCapture(filename)
        for i, frame_no in enumerate(i_frames):
            # Save I-Frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            outname = os.path.join(frame_path, f'i_frame_{i}.jpg')
            cv2.imwrite(outname, frame)

            # Save corresponding P-Frames until the next I-Frame
            next_i_frame = i_frames[i + 1] if (i + 1) < len(i_frames) else None
            p_frame_count = 0
            for p_frame_no in p_frames:
                if p_frame_no > frame_no and (next_i_frame is None or p_frame_no < next_i_frame):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, p_frame_no)
                    ret, frame = cap.read()
                    outname = os.path.join(frame_path, f'p_frame_{i}_{p_frame_count}.jpg')
                    cv2.imwrite(outname, frame)
                    p_frame_count += 1
                print('extracted')
        cap.release()
        print("Frame extraction Done!!")


# def extract_i_frames(video_path, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     try:
#         (
#             ffmpeg
#             .input(video_path)
#             .filter('select', 'eq(pict_type,I)')
#             .output(os.path.join(output_dir, 'i_frame_%04d.png'), vsync='vfr')
#             .run(capture_stdout=True, capture_stderr=True)
#         )
#     except ffmpeg.Error as e:
#         print('stdout:', e.stdout.decode('utf8'))
#         print('stderr:', e.stderr.decode('utf8'))
#         raise

# def extract_p_frames(video_path, output_dir, start_frame, end_frame):
#     os.makedirs(output_dir, exist_ok=True)
#     try:
#         select_filter = f'between(n\,{start_frame}\,{end_frame})'
#         (
#             ffmpeg
#             .input(video_path)
#             .filter('select', select_filter)
#             .output(os.path.join(output_dir, f'p_frame_{start_frame:04d}_%04d.png'), vsync='vfr')
#             .run(capture_stdout=True, capture_stderr=True)
#         )
#     except ffmpeg.Error as e:
#         print('stdout:', e.stdout.decode('utf8'))
#         print('stderr:', e.stderr.decode('utf8'))
#         raise

# def extract_i_p_frames(video_path, i_frame_dir, p_frame_dir):
#     os.makedirs(i_frame_dir, exist_ok=True)
#     os.makedirs(p_frame_dir, exist_ok=True)

#     # Extract I-frames
#     extract_i_frames(video_path, i_frame_dir)

#     # Get the list of extracted I-frames
#     i_frame_files = sorted(os.listdir(i_frame_dir))
#     i_frame_indices = []

#     # Parse the frame indices from the filenames
#     for file in i_frame_files:
#         index = int(file.split('_')[2].split('.')[0])
#         i_frame_indices.append(index)

#     # Extract P-frames for each I-frame range
#     for i in range(len(i_frame_indices) - 1):
#         start_frame = i_frame_indices[i]
#         end_frame = i_frame_indices[i + 1] - 1
#         extract_p_frames(video_path, p_frame_dir, start_frame, end_frame)

#     # Extract P-frames after the last I-frame
#     last_i_frame = i_frame_indices[-1]
#     extract_p_frames(video_path, p_frame_dir, last_i_frame, float('inf'))



# def extract_i_frames(video_path, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     try:
#         (
#             ffmpeg
#             .input(video_path)
#             .filter('select', 'eq(pict_type,I)')
#             .output(os.path.join(output_dir, 'i_frame_%04d.png'), vsync='vfr')
#             .run(capture_stdout=True, capture_stderr=True)
#         )
#     except ffmpeg.Error as e:
#         print('stdout:', e.stdout.decode('utf8'))
#         print('stderr:', e.stderr.decode('utf8'))
#         raise

# def extract_p_frames(video_path, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     try:
#         (
#             ffmpeg
#             .input(video_path)
#             .filter('select', 'eq(pict_type,P)')
#             .output(os.path.join(output_dir, 'p_frame_%04d.png'), vsync='vfr')
#             .run(capture_stdout=True, capture_stderr=True)
#         )
#     except ffmpeg.Error as e:
#         print('stdout:', e.stdout.decode('utf8'))
#         print('stderr:', e.stderr.decode('utf8'))
#         raise


#위 두함수로 통합시킴. 
# def extract_frames(video_path, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     ffmpeg.input(video_path).output(os.path.join(output_dir, 'frame_%04d.png')).run()

#v1
# def classify_frames(frame_dir, i_frame_dir, p_frame_dir):
#     os.makedirs(i_frame_dir, exist_ok=True)
#     os.makedirs(p_frame_dir, exist_ok=True)
#     # ffmpeg을 사용하여 I-frame과 P-frame을 분류하는 명령어를 실행
#     ffmpeg.input(frame_dir + '*.png').output(i_frame_dir + 'i_frame_%04d.png', vframes='1', force_key_frames='expr:gte(t,n_forced*2)').run()
#     ffmpeg.input(frame_dir + '*.png').output(p_frame_dir + 'p_frame_%04d.png', skip_frame='nokey').run()

#v2
# def classify_frames(frame_dir, i_frame_dir, p_frame_dir):
#     os.makedirs(i_frame_dir, exist_ok=True)
#     os.makedirs(p_frame_dir, exist_ok=True)
#     # ffmpeg을 사용하여 I-frame과 P-frame을 분류하는 명령어를 실행
#     input_pattern = os.path.join(frame_dir, 'frame_%04d.png')
#     i_frame_pattern = os.path.join(i_frame_dir, 'i_frame_%04d.png')
#     p_frame_pattern = os.path.join(p_frame_dir, 'p_frame_%04d.png')
    
#     (
#         ffmpeg
#         .input(input_pattern)
#         .output(i_frame_pattern, vframes='1', force_key_frames='expr:gte(t,n_forced*2)')
#         .run()
#     )
#     (
#         ffmpeg
#         .input(input_pattern)
#         .output(p_frame_pattern, skip_frame='nokey')
#         .run()
#     )
