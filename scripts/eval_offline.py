import sys
import json
import torch
import pathlib
import numpy as np
import dill
import argparse
import os
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra

# --- 1. Dynamic Path Setup ---
root_dir = pathlib.Path(__file__).resolve().parent.parent
package_root = root_dir / "packages" / "diffusion_policy" / "src"
sys.path.append(str(package_root))

from diffusion_policy.dataset.umi_dataset import UmiDataset
from diffusion_policy.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)

def load_split_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_eval_output_path(ckpt_path):
    path_obj = pathlib.Path(ckpt_path).resolve()
    parts = list(path_obj.parts)
    try:
        idx = parts.index("outputs")
        parts[idx] = "eval_results"
    except ValueError:
        return path_obj.parent / f"{path_obj.stem}_eval.json"
    return pathlib.Path(*parts).with_suffix('.json')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .ckpt file')
    parser.add_argument('--split', type=str, required=True, help='Path to split.json')
    parser.add_argument('--dataset', type=str, default=None, help='Override dataset path')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -------------------------------------------------------------------------
    # 1. Load Checkpoint & Metadata (Done Once)
    # -------------------------------------------------------------------------
    print(f"Loading checkpoint from {args.checkpoint}")
    payload = torch.load(open(args.checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    
    # Extract metadata
    train_epoch = -1
    train_global_step = -1
    if 'pickles' in payload:
        if 'epoch' in payload['pickles']:
            train_epoch = dill.loads(payload['pickles']['epoch'])
        if 'global_step' in payload['pickles']:
            train_global_step = dill.loads(payload['pickles']['global_step'])

    # -------------------------------------------------------------------------
    # 2. Dataset & Normalizer (Done Once)
    # -------------------------------------------------------------------------
    split_data = load_split_json(args.split)
    
    if args.dataset:
        target_dataset_path = args.dataset
    elif 'dataset' in split_data:
        target_dataset_path = split_data['dataset']
    else:
        target_dataset_path = cfg.task.dataset.dataset_path
        
    print(f"Loading dataset from {target_dataset_path}")
    train_dataset = UmiDataset(
        shape_meta=cfg.task.shape_meta,
        dataset_path=target_dataset_path,
        cache_dir=cfg.task.dataset.get('cache_dir', None),
        pose_repr=cfg.task.dataset.get('pose_repr', {}),
        action_padding=cfg.task.dataset.get('action_padding', False),
        temporally_independent_normalization=cfg.task.dataset.get('temporally_independent_normalization', False),
        seed=args.seed,
        val_ratio=0.0 
    )

    total_episodes = train_dataset.replay_buffer.n_episodes
    val_indices = split_data['val_episodes']

    new_val_mask = np.zeros(total_episodes, dtype=bool)
    new_val_mask[val_indices] = True
    train_dataset.val_mask = new_val_mask
    val_dataset = train_dataset.get_validation_dataset()

    print("Computing normalizer...")
    normalizer = train_dataset.get_normalizer()

    # -------------------------------------------------------------------------
    # 3. Define Evaluation Targets
    # -------------------------------------------------------------------------
    # We check which keys exist in the payload and build a list of tasks
    # Format: (JSON_Key, Payload_Key)
    eval_targets = []
    
    if 'state_dicts' in payload:
        state_dicts = payload['state_dicts']
        # if 'ema_model' in state_dicts and state_dicts['ema_model'] is not None:
        #     eval_targets.append(('ema_model', 'ema_model'))
        if 'model' in state_dicts:
            eval_targets.append(('model', 'model'))
        
        if not eval_targets:
             raise ValueError("No model weights (ema or model) found in state_dicts!")
    else:
        # Fallback for old checkpoints
        eval_targets.append(('model', 'state_dict_fallback'))

    final_results = {
        'epoch': train_epoch,
        'global_step': train_global_step,
        'data_root': target_dataset_path,
        'checkpoint': args.checkpoint
    }

    print(f"Models to evaluate: {[t[0] for t in eval_targets]}")

    # -------------------------------------------------------------------------
    # 4. Main Evaluation Loop
    # -------------------------------------------------------------------------
    # We loop here to ensure a clean policy initialization for every model type
    
    for output_key, internal_key in eval_targets:
        print(f"\n" + "="*50)
        print(f"Evaluating: {output_key}")
        print("="*50)
        
        # --- A. Re-Initialize Policy (Clean Slate) ---
        # This ensures no internal state (like device buffers) leaks between runs
        print("Initializing policy...")
        policy = hydra.utils.instantiate(cfg.policy)
        
        # --- B. Set Normalizer (On CPU) ---
        policy.set_normalizer(normalizer)
        
        # --- C. Load Weights ---
        if internal_key == 'state_dict_fallback':
            print("Loading legacy state_dict...")
            policy.load_state_dict(payload['state_dict'], strict=True)
        else:
            print(f"Loading weights from state_dicts['{internal_key}']...")
            policy.load_state_dict(payload['state_dicts'][internal_key], strict=True)
            
        # --- D. Move to Device ---
        device = torch.device(args.device)
        policy.to(device)
        policy.eval()
        
        # --- E. Create DataLoader ---
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=(args.num_workers > 0)
        )
        
        # --- F. Run Eval ---
        mse_accum = {'all': 0.0, 'pos': 0.0, 'rot': 0.0, 'width': 0.0}
        count = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Running {output_key}"):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                
                gt_action = batch['action']
                result = policy.predict_action(batch['obs'])
                
                if 'action_pred' in result:
                    pred_action = result['action_pred']
                else:
                    pred_action = result['action']
                
                # Metrics
                B, T, D = pred_action.shape
                loss_all = torch.nn.functional.mse_loss(pred_action, gt_action, reduction='sum')
                loss_pos = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3], reduction='sum')
                loss_rot = torch.nn.functional.mse_loss(pred_action[..., 3:9], gt_action[..., 3:9], reduction='sum')
                loss_width = torch.nn.functional.mse_loss(pred_action[..., 9], gt_action[..., 9], reduction='sum')

                mse_accum['all'] += loss_all.item() / D
                mse_accum['pos'] += loss_pos.item() / 3
                mse_accum['rot'] += loss_rot.item() / 6
                mse_accum['width'] += loss_width.item() / 1
                count += (B * T)
        
        # Store results
        final_results[output_key] = {
            'val/action_mse_error': mse_accum['all'] / count,
            'val/action_mse_error_pos': mse_accum['pos'] / count,
            'val/action_mse_error_rot': mse_accum['rot'] / count,
            'val/action_mse_error_width': mse_accum['width'] / count,
        }
        
        print(f"Result ({output_key}): {final_results[output_key]['val/action_mse_error']:.6f}")
        
        # Clean up to free memory for next iteration
        del policy
        del val_loader
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 5. Save Results
    # -------------------------------------------------------------------------
    print("-" * 50)
    output_path = get_eval_output_path(args.checkpoint)
    print(f"Saving combined results to: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    main()