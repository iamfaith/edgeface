#!/usr/bin/env python3
"""
Export a model from `backbones.get_model` to ONNX.

Usage example:
  python3 export_to_onnx.py --model edgeface_s_gamma_05 --checkpoint checkpoints/edgeface_s_gamma_05.pt --output checkpoints/edgeface_s_gamma_05.onnx

The script attempts to be robust to checkpoints that store a dict with key 'state_dict'.
"""
import argparse
import os
import sys
import torch


def load_model_and_weights(model_name, checkpoint_path, device='cpu'):
    # import here to ensure repository modules are on sys.path
    from backbones import get_model

    model = get_model(model_name)

    state = torch.load(checkpoint_path, map_location=device)
    # support checkpoint formats that wrap the state dict
    if isinstance(state, dict):
        if 'state_dict' in state:
            state = state['state_dict']
        # some checkpoints prefix keys with 'module.' from DataParallel
        # let load_state_dict handle mismatches, but try to strip common prefix
        if any(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', ''): v for k, v in state.items()}

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def export_onnx(model, output_path, input_size=(1, 3, 112, 112), opset=13, dynamic_batch=True, device='cpu'):
    dummy = torch.randn(*input_size, device=device)
    input_names = ['input']
    output_names = ['output']
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )


def parse_args():
    p = argparse.ArgumentParser(description='Export model (from backbones.get_model) to ONNX')
    p.add_argument('--model', help='model name for get_model(), e.g. edgeface_s_gamma_05', default='edgeface_s_gamma_05')
    p.add_argument('--checkpoint', help='path to checkpoint .pt file', default='/home/faith/edgeface/checkpoints/edgeface_s_gamma_05.pt')
    p.add_argument('--output', default=None, help='output ONNX path (default: <model>.onnx)')
    p.add_argument('--batch', type=int, default=1, help='dummy batch size')
    p.add_argument('--height', type=int, default=112, help='input height')
    p.add_argument('--width', type=int, default=112, help='input width')
    p.add_argument('--opset', type=int, default=13, help='ONNX opset version (11/12/13 recommended)')
    p.add_argument('--cpu', action='store_true', help='force CPU instead of CUDA')
    p.add_argument('--no-dyn', action='store_true', help='disable dynamic batch size in ONNX')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')

    if args.output:
        out = args.output
    else:
        out = f"{args.model}.onnx"

    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(2)

    print(f"Loading model `{args.model}` from `{args.checkpoint}` -> device={device}")
    model = load_model_and_weights(args.model, args.checkpoint, device=device)

    input_size = (args.batch, 3, args.height, args.width)
    print(f"Exporting to {out} (opset={args.opset}, input_size={input_size}, dynamic_batch={not args.no_dyn})")

    try:
        export_onnx(model, out, input_size=input_size, opset=args.opset, dynamic_batch=(not args.no_dyn), device=device)
    except Exception as e:
        print('ONNX export failed:', e)
        print('Hints: try a different --opset (e.g. 11 or 12), run with --cpu, or add --no-dyn to disable dynamic axes.')
        raise

    print('ONNX export finished:', out)


if __name__ == '__main__':
    main()
