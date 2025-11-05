import os
from pathlib import Path
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from src.dataset.avhubert_dataset import AudioTransform, VideoTransform, DataCollator
from src.tokenizer.spm_tokenizer import TextTransform
from src.avhubert_avsr.avhubert_avsr_model import AVHubertAVSR, get_beam_search_decoder
from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
from src.talking_detector.segmentation import segment_by_asd
from script.lip_crop import LazyVideo
from datasets import load_from_disk
import torch, torchvision, torchaudio
from src.cluster.conv_spks import (
    get_speaker_activity_segments,
    calculate_conversation_scores,
    cluster_speakers,
    get_clustering_f1_score
)
from torchcodec.decoders import VideoDecoder
import json
import math
from tqdm import tqdm
import glob
import warnings

from transformers import AutoImageProcessor, AutoModel
from script.dino_inference import read_video_frames, batched
from torch import nn

FPS = 25

class Dino(nn.Module):
    def __init__(self, model_id, device, batch_size=64):
        super().__init__()

        print(f"[INFO] Loading DINOv3 model: {model_id}")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available())

        self.model.eval().to(device)
        self.device = device
        self.batch_size = batch_size

    def __call__(self, video_path):
        lv = LazyVideo(video_path)
        if len(lv) == 0:
            raise RuntimeError("No frames decoded from the video.")

        all_embeds = []
        with torch.no_grad(), self.autocast_ctx:
            for batch in tqdm(batched(lv, self.batch_size), total=len(lv) // self.batch_size):
                if batch.sum() == 0 and len(all_embeds) > 0:
                    all_embeds.extend(torch.zeros((len(batch), len(all_embeds[-1])), dtype=all_embeds[-1].dtype, device=all_embeds[-1].device))
                    continue

                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                # Prefer pooled embedding if available; else use CLS token (index 0)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output  # [B, D]
                else:
                    # last_hidden_state: [B, seq_len, D] -> take CLS at position 0
                    emb = outputs.last_hidden_state[:, 0, :]  # [B, D]

                embs = emb.detach().to("cpu", dtype=torch.float32)
                if batch.sum() == 0:
                    all_embeds.extend(torch.zeros_like(embs))
                    continue
                else:
                    zero_idxes = batch.reshape(batch.shape[0], -1).sum(axis=-1) == 0
                    embs[zero_idxes].zero_()

                # Move to CPU in float32 for consistent saving
                all_embeds.extend(embs)

        return torch.stack(all_embeds).unsqueeze(0).unsqueeze(2)


def load_model(is_dino: bool = False, device: str = "cpu"):
    # Load text transform
    sp_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram/unigram5000.model")
    dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram/unigram5000_units.txt")
    text_transform = TextTransform(
        sp_model_path=sp_model_path,
        dict_path=dict_path,
    )

    # Load data collator
    audio_transform = AudioTransform(subset="test")
    video_transform = VideoTransform(subset="test")

    av_data_collator = DataCollator(
        text_transform=text_transform,
        audio_transform=audio_transform,
        video_transform=video_transform,
    )

    # Load model
    if is_dino:
        model = Dino(model_id="facebook/dinov3-vits16-pretrain-lvd1689m", device=device)
        # model = Dino(model_id="facebook/dinov3-vit7b16-pretrain-lvd1689m", device=device)
        # model = Dino(model_id="facebook/dinov3-vitl16-pretrain-lvd1689m", device=device)
    else:
        model_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model-bin/avsr_cocktail")
        avsr_model = AVHubertAVSR.from_pretrained(model_name)
        avsr_model.eval()
        model = avsr_model.avsr

    return model, text_transform, av_data_collator


def inference(model, video, audio, embed_source, layers):
    avhubert_features = model.encoder(
        input_features = audio,
        video = video,
        embed_source = embed_source,
        layers = layers,
    )
    audiovisual_feat = avhubert_features.last_hidden_state
    return audiovisual_feat

def chunk_video(video_path, asd_path=None, max_length=10):
    # load video and split into chunks for inference
    if asd_path is not None:
        with open(asd_path, "r") as f:
            asd = json.load(f)

        # Convert frame numbers to integers and sort them
        frames = sorted([int(f) for f in asd.keys()])
        # Find the minimum frame number to normalize frame indices
        min_frame = min(frames)

        segments_by_frames = segment_by_asd(asd, {
            "max_chunk_size": max_length,  # in seconds
        })
        # Normalize frame indices, for inference, don't care about the actual frame indices
        segments = [((seg[0] - min_frame) / FPS, (seg[-1] - min_frame) / FPS) for seg in segments_by_frames]

    else:
        # # Get video duration
        # audio, rate = torchaudio.load(video_path)
        # # This is not correct - the corresponding video has different sampling rate (FPSfps).
        # # Therefore, we need to fix the length and account for the actual fps instead of audio sampling rate.
        # audio_len = audio.shape[1]
        # audio_video_downsample_factor = 640 # 16_000 / FPS
        # num_vid_frames = audio_len / audio_video_downsample_factor
        # video_duration = (audio_len - (num_vid_frames - int(num_vid_frames)) * audio_video_downsample_factor) / rate
        # num chunks
        video_duration = len(VideoDecoder(video_path)) / FPS # video_framerate
        num_chunks = math.ceil(video_duration / max_length)
        chunk_size = math.ceil(video_duration / num_chunks)
        segments = []
        # Convert to integer steps for range
        steps = int(video_duration * 100)  # Convert to centiseconds for precision
        step_size = int(chunk_size * 100)
        for i in range(0, steps, step_size):
            start_time = i / 100
            # -0.2 due to a rounding error causing exceptions when loading videos
            end_time = min((i + step_size) / 100, video_duration)
            assert start_time < end_time
            segments.append((start_time, end_time))

    # if len(segments) > 0:
    #     print(f"Total segments: {len(segments)}")
    #     for idx, seg in enumerate(segments):
    #         print(f"Segment {idx}: {seg[0]:.2f}s - {seg[-1]:.2f}s, len: {seg[-1] - seg[0]:.2f}s")
    #     print(f"Max segment length: {max([seg[-1] - seg[0] for seg in segments]):.2f}s")
    #     print(f"Min segment length: {min([seg[-1] - seg[0] for seg in segments]):.2f}s")
    return segments


def format_vtt_timestamp(timestamp):
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) // 60)
    seconds = int(timestamp % 60)
    milliseconds = int((timestamp - int(timestamp)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def infer_video(
    model,
    av_data_collator,
    video_path, embed_source, layers, device, asd_path=None, offset=0.
):
    if embed_source == 'dinov3':
        video_duration = len(VideoDecoder(video_path)) / FPS
        return [{
            "start_time": offset,
            "end_time": offset + video_duration,
            "feats": model(video_path)[0]
        }]
    # asd_path = None -> equal-length segments
    asd_path=None
    segments = chunk_video(video_path, asd_path, max_length=10)
    segment_output = []
    for seg in tqdm(segments, desc="Processing segments", total=len(segments)):
        # Inference
        sample = {
            "video": video_path,
            "start_time": seg[0],
            "end_time": seg[1],
        }

        sample_features = av_data_collator([sample])
        audios = sample_features["audios"]
        videos = sample_features["videos"]
        audio_lengths = sample_features["audio_lengths"]
        video_lengths = sample_features["video_lengths"]
        with torch.no_grad():
            segment_feats = inference(model, videos.to(device), audios.to(device), embed_source, layers).cpu()
            segment_output.extend(list(segment_feats))

    return [
        {
            "start_time": seg[0] + offset,
            "end_time": seg[1] + offset,
            "feats": output
        } for seg, output in zip(segments, segment_output)
    ]

def mcorec_session_infer(model, av_data_collator, session_dir, output_dir: Path, save_mode, fill_gaps_method, embed_source, layers, device):
    # Infer session
    with open(os.path.join(session_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # Tracks can be shorter and end sooner than the overall video.
    # We want to make sure that if we load the central audio from the central video, the durations will match.
    # Hence, we need to pad speaker-specific feats to this # of vid. frames.
    total_num_frames = len(VideoDecoder(session_dir / "central_video.mp4"))

    # Process speaker transcripts
    for speaker_name, speaker_data in tqdm(list(metadata.items())[::-1], desc="Processing speakers", total=len(metadata)):
        speaker_feats = []
        spk_output_dir = output_dir / speaker_name
        os.makedirs(spk_output_dir, exist_ok=True)

        # Check all the crops and select only the unique ones.
        # session_60 has duplicated tracks.
        unique_tracks = []
        unique_crop_metadata = set()
        for track in speaker_data['central']['crops']:
            with open(os.path.join(session_dir, track['crop_metadata']), "r") as f:
                crop_metadata = json.load(f)
            track['crop_metadata'] = crop_metadata
            key = (crop_metadata['start_time'], crop_metadata['end_time'])
            if key not in unique_crop_metadata:
                unique_crop_metadata.add(key)
                unique_tracks.append(track)

        if len(unique_tracks) != len(speaker_data['central']['crops']):
            warnings.warn(f'The number of unique tracks is different from the total number of tracks. '
                          f'Duplicated tracks will be ignored: session: {os.path.basename(session_dir)}, speaker: {speaker_name}')

        # for track in speaker_data['central']['crops']:
        for track in unique_tracks:
            video_path = os.path.join(session_dir, track['lip'])
            asd_path = os.path.join(session_dir, track['asd']) if 'asd' in track else None
            # with open(os.path.join(session_dir, track['crop_metadata']), "r") as f:
            #     crop_metadata = json.load(f)
            crop_metadata = track['crop_metadata']
            track_start_time = crop_metadata['start_time']
            speaker_feats.extend(infer_video(
                model,
                av_data_collator,
                video_path,
                embed_source=embed_source,
                layers=layers,
                device=device,
                asd_path=asd_path,
                offset=track_start_time))

        speaker_feats.sort(key=lambda x: x['start_time'])

        if save_mode == 'per_speaker_tracks_combined':
            # # of frames we need to fill.
            gaps = [0] + [round((x['start_time'] - y['end_time']) * FPS) for x, y in
                          zip(speaker_feats[1:], speaker_feats[:-1])]
            filled_in_feat_sequence = []
            for segment, gap_before in zip(speaker_feats, gaps):
                if gap_before > 0:
                    if fill_gaps_method == 'fill_zeros':
                        filled_in_feat_sequence.append(torch.zeros(gap_before, *segment['feats'].shape[-2:]))
                filled_in_feat_sequence.append(segment['feats'])

            concat_seq = torch.concat(filled_in_feat_sequence, dim=0)
            assert concat_seq.shape[0] / FPS - speaker_feats[-1]['end_time'] < 1e-3, f'END TIMES DO NOT MATCH: {concat_seq.shape[0] / FPS - speaker_feats[-1]["end_time"]}'

            n_frames_diff = total_num_frames - concat_seq.shape[0]
            assert n_frames_diff >= 0
            if n_frames_diff > 0:
                concat_seq = torch.concat((concat_seq, torch.zeros((n_frames_diff, *concat_seq.shape[-2:]))), dim=0)

            assert concat_seq.shape[0] == total_num_frames
            torch.save(concat_seq, spk_output_dir / "all_tracks.pt")
        else:
            raise NotImplementedError

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inferring speaker clustering and transcripts from video")
    parser.add_argument('--session_dir', type=str, required=True, help='Path to folder containing session data')
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_mode', type=str,
                        choices=['per_track', 'per_speaker_tracks_combined', 'both'],
                        default='per_speaker_tracks_combined',
                        help='''
                        per_track - one sequence of tensors per one track,
                        per_speaker_tracks_combined - concat tracks with using `fill_gaps_method`,
                        both - performs the both previously mentioned methods.
                        ''')
    parser.add_argument('--fill_gaps_method', type=str,
                        choices=['fill_zeros'],
                        default='fill_zeros',
                        help='fill_zeros - fill track gaps with zero tensors of the same dimension and frame-rate as the video features (25fps).')
    parser.add_argument('--embed_source',
                        choices=['av', 'av_v_only', 'vision_only', 'dinov3'],
                        default='av_last_layer',
                        help='av - both modalities are passed to encoder, features are taken from the multi-modal tarnsformer encoder, '
                             'av_v_only - only video is passed to encoder, audio is set to 0, features are taken from the multi-modal tarnsformer encoder, '
                             'vision_only - embeddings extracted by the vision encoder (before av transformer encoder) - ResNet.'
                        )
    parser.add_argument('--layers', type=str, default="-1", help='Either a number, -1 - last, all - return all layers stacked in one tensor.')

    opt = parser.parse_args()

    output_dir = Path(opt.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # If we have a safe_gpu installed, we use it to select a free GPU.
    try:
        from safe_gpu import safe_gpu
        from time import sleep
        while True:
            try:
                safe_gpu.claim_gpus(1)
                break
            except:
                sleep(1)
    except:
        pass

    device = torch.device(opt.device)

    # Load model
    model, text_transform, av_data_collator = load_model(is_dino=opt.embed_source == 'dinov3', device=device)
    model = model.to(device)

    if opt.session_dir.strip().endswith("*"):
        all_session_dirs = glob.glob(opt.session_dir)
    else:
        all_session_dirs = [opt.session_dir]
    print(f"Infering {len(all_session_dirs)} sessions")

    for session_dir in all_session_dirs:
        session_output_dir = output_dir / os.path.basename(session_dir)
        os.makedirs(session_output_dir, exist_ok=True)
        print(f"Infering session {session_dir.split('/')[-1]}")
        mcorec_session_infer(model, av_data_collator,
                             session_dir=Path(session_dir),
                             output_dir=session_output_dir,
                             save_mode=opt.save_mode,
                             fill_gaps_method=opt.fill_gaps_method,
                             embed_source=opt.embed_source,
                             layers=opt.layers,
                             device=device)

if __name__ == "__main__":
    main()


