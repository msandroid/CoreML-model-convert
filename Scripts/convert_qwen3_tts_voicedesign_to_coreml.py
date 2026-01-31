#!/usr/bin/env python3
"""
Convert Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign to Core ML (.mlpackage).

Pipeline: Text + Instruct -> Tokenize -> Talker (LM, autoregressive) -> SpeechTokenizer decode -> Waveform.
The Talker uses autoregressive generation; conversion focuses on extractable components.
Speech tokenizer decode may be traceable.

Usage:
  python Scripts/convert_qwen3_tts_voicedesign_to_coreml.py --output-dir ./models/qwen3-tts-voicedesign/coreml
  python Scripts/convert_qwen3_tts_voicedesign_to_coreml.py --model-dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign --output-dir ./models/qwen3-tts-voicedesign/coreml --variants 4bit 5bit 6bit 8bit fp16

Requires: pip install qwen-tts torch coremltools
See: Docs/Qwen3-TTS-VoiceDesign-CoreML-Conversion.md
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
VARIANTS = ["4bit", "5bit", "6bit", "8bit", "fp16"]

# Variant -> nbits for quantization_utils.quantize_weights (neural network format)
# fp16: no quantization, use compute_precision
VARIANT_NBITS = {
    "4bit": 4,
    "5bit": 5,
    "6bit": 6,
    "8bit": 8,
    "fp16": None,
}


def _load_model(model_path: str, attn_implementation: str = "eager"):
    """Load Qwen3-TTS VoiceDesign model via qwen-tts.
    Use attn_implementation='eager' to avoid SDPA causal masking that fails JIT trace.
    """
    from qwen_tts import Qwen3TTSModel

    load_kwargs = {"torch_dtype": torch.float32}
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    model = Qwen3TTSModel.from_pretrained(model_path, **load_kwargs)
    model.model.eval()
    return model


def _copy_tokenizer_files(model_dir: Path, out_dir: Path):
    """Copy tokenizer and config files to output directory."""
    files_to_copy = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
    ]
    tokenizer_out = out_dir / "tokenizer"
    tokenizer_out.mkdir(parents=True, exist_ok=True)
    for fname in files_to_copy:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, tokenizer_out / fname)
            print(f"  Copied {fname}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS-12Hz-1.7B-VoiceDesign to Core ML"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Local directory with config.json and safetensors (skips download)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=REPO_ID,
        help="Hugging Face repo id when not using --model-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for .mlpackage files (e.g. ./models/qwen3-tts-voicedesign/coreml)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=VARIANTS,
        choices=VARIANTS,
        help="Output variants to create (default: all 4bit 5bit 6bit 8bit fp16)",
    )
    parser.add_argument(
        "--minimum-ios-deployment-target",
        type=str,
        default="18",
        help="Minimum iOS version (default 18)",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        help="Attention implementation: eager, sdpa, flash_attention_2, or none (default: eager)",
    )
    parser.add_argument(
        "--skip-speech-decoder",
        action="store_true",
        help="Skip speech decoder conversion (copy tokenizer only). Use when RAM is limited.",
    )
    args = parser.parse_args()

    model_path = args.model_dir or args.repo_id
    model_dir = Path(model_path)
    base_out = Path(args.output_dir).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    model = None
    if not args.skip_speech_decoder:
        attn_impl = None if (args.attn_implementation or "").lower() == "none" else (args.attn_implementation or "eager")
        print("Loading model...", file=sys.stderr)
        try:
            qwen_model = _load_model(model_path, attn_implementation=attn_impl)
        except TypeError as te:
            print(f"attn_implementation not supported, retrying without: {te}", file=sys.stderr)
            qwen_model = _load_model(model_path, attn_implementation=None)
        except Exception as e:
            print(f"Model load failed: {e}", file=sys.stderr)
            print("Install: pip install qwen-tts torch", file=sys.stderr)
            print("Download: huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign", file=sys.stderr)
            return 1
        model = qwen_model.model
    else:
        print("Skipping model load (--skip-speech-decoder). Copying tokenizer only.", file=sys.stderr)

    try:
        import coremltools as ct
    except ImportError:
        print("Error: coremltools required. pip install coremltools", file=sys.stderr)
        return 1

    target_map = {
        "13": ct.target.iOS13,
        "14": ct.target.iOS14,
        "15": ct.target.iOS15,
        "16": ct.target.iOS16,
        "17": ct.target.iOS17,
        "18": ct.target.iOS18,
    }
    target = target_map.get(args.minimum_ios_deployment_target.strip(), ct.target.iOS18)

    # Inspect model structure
    if model is not None:
        print("Model structure:", file=sys.stderr)
        for name, _ in model.named_children():
            print(f"  - {name}", file=sys.stderr)

    # Copy tokenizer files for each variant
    for variant in args.variants:
        variant_out = base_out / variant
        variant_out.mkdir(parents=True, exist_ok=True)
        if model_dir.exists():
            _copy_tokenizer_files(model_dir, variant_out)
        else:
            print(f"  Warning: --model-dir not set; tokenizer files may need manual copy to {variant_out}", file=sys.stderr)

    # Attempt to convert speech_tokenizer decoder (st.model.decoder)
    if model is not None and hasattr(model, "speech_tokenizer"):
        st = model.speech_tokenizer
        decoder = getattr(getattr(st, "model", None), "decoder", None) if hasattr(st, "model") else getattr(st, "decoder", None)
        if decoder is not None and hasattr(decoder, "forward"):
            try:
                print("Attempting speech_tokenizer decoder conversion...", file=sys.stderr)
                # Monkey-patch create_causal_mask for JIT trace: qwen_tts imports it from transformers;
                # the original uses packed_sequence_mask with vmap that fails. Patch in qwen_tts namespace.
                import qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 as qwen_tok

                _original_create_causal_mask = qwen_tok.create_causal_mask
                _original_create_sliding = qwen_tok.create_sliding_window_causal_mask

                def _traceable_create_causal_mask(**kwargs):
                    emb = kwargs.get("input_embeds")
                    if emb is not None:
                        b, q_len, _ = emb.shape
                        cache_pos = kwargs.get("cache_position")
                        kv_len = cache_pos.shape[0] if cache_pos is not None and hasattr(cache_pos, "shape") else q_len
                        causal = torch.tril(
                            torch.ones((q_len, kv_len), dtype=emb.dtype, device=emb.device)
                        ).view(1, 1, q_len, kv_len)
                        return causal
                    return _original_create_causal_mask(**kwargs)

                def _traceable_create_sliding(**kwargs):
                    return _traceable_create_causal_mask(**kwargs)

                qwen_tok.create_causal_mask = _traceable_create_causal_mask
                qwen_tok.create_sliding_window_causal_mask = _traceable_create_sliding
                try:
                    # Qwen3 12Hz: 16 codebook layers, shape (batch, n_layers, seq_len)
                    batch, n_layers, seq_len = 1, 16, 100
                    dummy_codes = torch.randint(0, 2048, (batch, n_layers, seq_len), dtype=torch.int64)
                    with torch.no_grad():
                        out = decoder(dummy_codes)
                    if not isinstance(out, torch.Tensor):
                        raise ValueError("Decoder output is not a tensor")
                    traced = torch.jit.trace(decoder, (dummy_codes,))
                    # Use neuralnetwork format for quantize_weights compatibility (4,5,6,8 bit)
                    base_mlmodel = ct.convert(
                        traced,
                        source="pytorch",
                        inputs=[
                            ct.TensorType(
                                name="audio_codes",
                                shape=(batch, n_layers, seq_len),
                                dtype=np.int64,
                            )
                        ],
                        minimum_deployment_target=target,
                        convert_to="neuralnetwork",
                    )
                    for variant in args.variants:
                        variant_out = base_out / variant
                        variant_out.mkdir(parents=True, exist_ok=True)
                        out_path = variant_out / "qwen3_tts_speech_decoder.mlpackage"
                        nbits = VARIANT_NBITS.get(variant)
                        if nbits is not None:
                            try:
                                from coremltools.models.neural_network import quantization_utils
                                qmodel = quantization_utils.quantize_weights(
                                    base_mlmodel, nbits=nbits, quantization_mode="linear"
                                )
                                qmodel.save(str(out_path))
                            except Exception as qe:
                                print(f"Quantization {variant} ({nbits}bit) failed: {qe}", file=sys.stderr)
                                base_mlmodel.save(str(out_path))
                                print(f"Saved unquantized to {out_path}", file=sys.stderr)
                        else:
                            # fp16: use FP16 quantization
                            try:
                                from coremltools.models.neural_network import quantization_utils
                                fp16_model = quantization_utils.quantize_weights(
                                    base_mlmodel, nbits=16, quantization_mode="linear"
                                )
                                fp16_model.save(str(out_path))
                            except Exception as fe:
                                print(f"FP16 conversion failed: {fe}", file=sys.stderr)
                                base_mlmodel.save(str(out_path))
                        print(f"Saved {out_path}", file=sys.stderr)
                finally:
                    qwen_tok.create_causal_mask = _original_create_causal_mask
                    qwen_tok.create_sliding_window_causal_mask = _original_create_sliding
            except Exception as e:
                print(f"Speech decoder conversion failed: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                print("The speech_tokenizer may use a different structure. Manual inspection needed.", file=sys.stderr)

    # Note: The main Talker/LM uses autoregressive model.generate() which is not easily traceable.
    # A full conversion would require extracting the decoder's single-step forward and running
    # the generation loop in Swift. This is documented for future implementation.
    print("", file=sys.stderr)
    print("Note: The Talker (LM) uses autoregressive generation. CoreML conversion of the", file=sys.stderr)
    print("main model requires extracting single-step forward and implementing the loop in Swift.", file=sys.stderr)
    print("See Docs/Qwen3-TTS-VoiceDesign-CoreML-Conversion.md for details.", file=sys.stderr)
    print("", file=sys.stderr)
    print("Done. Output in:", base_out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
