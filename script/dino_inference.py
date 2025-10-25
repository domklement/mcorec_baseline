import argparse
import os
from typing import List, Tuple

import cv2
import torch
from transformers import AutoImageProcessor, AutoModel

# -----------------------------
# Utilities
# -----------------------------
def read_video_frames(
    video_path: str,
    frame_stride: int = 1,
    max_frames: int = None
) -> Tuple[List, List[int]]:
    """
    Decode video and return a list of RGB frames (numpy arrays) and their indices.
    frame_stride=1 -> every frame, 2 -> every other frame, etc.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_rgb = []
    kept_indices = []

    idx = 0
    kept = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if idx % frame_stride == 0:
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
            kept_indices.append(idx)
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
        idx += 1

    cap.release()
    return frames_rgb, kept_indices


def batched(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract per-frame DINOv3 embeddings from a video and save to a .pt tensor."
    )
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Hugging Face model id (DINOv3). Examples: "
             "'facebook/dinov3-vitb16-pretrain-lvd1689m', "
             "'facebook/dinov3-vit7b16-pretrain-lvd1689m'."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pt path (defaults to <video_stem>__dinov3.pt)"
    )
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Keep every Nth frame (1 = all frames).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional cap on number of frames processed (after stride).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of frames per forward pass.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on: 'cuda' or 'cpu'.")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Computation dtype for the model inputs.")
    args = parser.parse_args()

    # facebook/dinov3-vits16-pretrain-lvd1689m

    video_path = args.video_path
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    out_path = args.output or (
        os.path.splitext(video_path)[0] + "__dinov3.pt"
    )

    print(f"[INFO] Loading DINOv3 model: {args.model_id}")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval().to(args.device)

    # mixed precision context
    amp_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype) if args.device.startswith("cuda")
        else torch.cpu.amp.autocast(dtype=amp_dtype)
    )

    print(f"[INFO] Decoding video and sampling frames (stride={args.frame_stride})...")
    frames, frame_indices = read_video_frames(
        video_path, frame_stride=args.frame_stride, max_frames=args.max_frames
    )
    if len(frames) == 0:
        raise RuntimeError("No frames decoded from the video.")

    # We'll collect per-frame global embeddings (pooled output / CLS token)
    # DINOv3 in HF exposes last_hidden_state; the pooled output is in `pooler_output` if provided.
    all_embs = []
    with torch.no_grad(), autocast_ctx:
        for batch in batched(frames, args.batch_size):
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(args.device) for k, v in inputs.items()}

            outputs = model(**inputs)

            # Prefer pooled embedding if available; else use CLS token (index 0)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output  # [B, D]
            else:
                # last_hidden_state: [B, seq_len, D] -> take CLS at position 0
                emb = outputs.last_hidden_state[:, 0, :]  # [B, D]

            # Move to CPU in float32 for consistent saving
            all_embs.append(emb.detach().to("cpu", dtype=torch.float32))

    embeddings = torch.cat(all_embs, dim=0)  # [num_frames, D]

    # Save a dict so we can keep frame indices & model metadata alongside the tensor
    payload = {
        "embeddings": embeddings,              # torch.Size([N, D])
        "frame_indices": torch.tensor(frame_indices, dtype=torch.int32),
        "model_id": args.model_id,
        "dtype": "float32",
        "frame_stride": args.frame_stride,
        "video_path": os.path.abspath(video_path),
    }
    torch.save(payload, out_path)
    print(f"[DONE] Saved embeddings: {embeddings.shape} -> {out_path}")


if __name__ == "__main__":
    main()