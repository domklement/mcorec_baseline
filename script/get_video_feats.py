import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from script.get_feats import load_model, infer_video

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inferring speaker clustering and transcripts from video")
    parser.add_argument('--input_video', type=str, required=True, help='Path to folder containing session data')
    parser.add_argument('--output_file', required=True, type=str)
    parser.add_argument('--device', type=str, default='cpu')
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

    if os.path.dirname(opt.output_file):
        os.makedirs(os.path.dirname(opt.output_file), exist_ok=True)

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

    # infer video directly, import as many funcs from get_feats as possible. This should be rather a short script.
    try:
        feats = infer_video(
            model,
            av_data_collator,
            opt.input_video,
            embed_source=opt.embed_source,
            layers=opt.layers,
            device=device)

        torch.save(feats[0]['feats'], opt.output_file)
    except Exception as e:
        print(f"Error processing {opt.input_video}")
        raise e


if __name__ == "__main__":
    main()


