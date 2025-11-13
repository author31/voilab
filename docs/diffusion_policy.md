# Diffusion Policy

This diffusion policy package is a modified version of the original diffusion policy package. It is designed to be used with the UMI-collected dataset.

## Training

To train the diffusion policy, you can use the following command:

```bash
uv run packages/diffusion_policy/train.py --config-path=src/diffusion_policy/config --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=/path/to/your/dataset.zarr.zip
```
