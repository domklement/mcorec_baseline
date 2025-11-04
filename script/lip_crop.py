import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retinaface.detector import LandmarksDetector
from src.retinaface.video_process import VideoProcess
from src.retinaface.utils import save_vid_aud_txt, save2vid, save2aud
from tqdm import tqdm
import traceback
import numpy as np
import math

import torch
import torchaudio
import torchvision
import json
import ffmpeg
import glob
import cv2

FACE_CROP_SIZE=224
FACE_CROP_MARGIN=12
FPS=25

# ==================== LOAD MODEL ====================

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
landmarks_detector = LandmarksDetector(device=device)
video_process = VideoProcess(convert_gray=False)
face_video_process = VideoProcess(crop_width=FACE_CROP_SIZE,
                                  crop_height=FACE_CROP_SIZE,
                                  start_idx=0,
                                  stop_idx=68,
                                  convert_gray=False,
                                  window_margin=FACE_CROP_MARGIN)

def process_video(video_path, output_dir=None, process_audio=True):
    try:
        if process_audio:
            # Load and process audio and video
            audio, sample_rate = torchaudio.load(video_path, normalize=True)

        video, _, meta_info = torchvision.io.read_video(video_path)
        video = video.numpy()
        assert meta_info['video_fps'] == FPS
        landmarks = landmarks_detector(video, output_face_bboxes=False)

        face_segment_name = video_path.split("/")[-1].replace(".mp4", "")
        segment_name = video_path.split("/")[-1].replace(".mp4", "_lip")
        if output_dir is None:
            output_dir = os.path.dirname(video_path)
        os.makedirs(output_dir, exist_ok=True)

        face_crop_video = face_video_process(video, landmarks)
        video = video_process(video, landmarks)
        video = torch.tensor(video)

        dst_vid_filename = os.path.join(output_dir, f"{segment_name}.mp4")
        save2vid(os.path.join(output_dir, f"{face_segment_name}.mp4"), face_crop_video, FPS)
        save2vid(dst_vid_filename, video, FPS)

        if process_audio:
            dst_aud_filename = os.path.join(output_dir, f"{segment_name}.wav")
            save2aud(dst_aud_filename, audio, sample_rate)

        # text_filename = os.path.join(output_dir, f"{segment_name}.json")
        # save_vid_aud_txt(
        #     dst_vid_filename,
        #     None if not process_audio else dst_aud_filename,
        #     text_filename,
        #     video,
        #     None if not process_audio else audio,
        #     json.dumps({
        #         "path": video_path
        #     }, indent=4),
        #     video_fps=25,
        #     audio_sample_rate=16000,
        # )

        # if process_audio:
        #     # Combine audio and video
        #     in1 = ffmpeg.input(dst_vid_filename)
        #     in2 = ffmpeg.input(dst_aud_filename)
        #     out = ffmpeg.output(
        #         in1["v"],
        #         in2["a"],
        #         dst_vid_filename[:-4] + ".av.mp4",
        #         vcodec="copy",
        #         acodec="aac",
        #         strict="experimental",
        #         loglevel="panic",
        #     )
        #     out.run(overwrite_output=True)
    except Exception as e:
        traceback.print_exc()
        # print(f"Error processing {video_path} segment {segment_frame[0]}-{segment_frame[-1]}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Active Speaker Detection")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--process_audio', action='store_true')
    opt = parser.parse_args()

    process_video(opt.video, output_dir=opt.output_dir, process_audio=opt.process_audio)


if __name__ == "__main__":
    main()