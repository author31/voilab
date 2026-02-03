import copy

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.common.pytorch_util import replace_submodules

logger = logging.getLogger(__name__)

# ============================================================
# Utilities
# ============================================================

def compute_vit_ray_embedding(H, W, patch_size, K, device):
    """
    Return ray embedding for ViT patch tokens
    Shape: (1 + N, 3)  (CLS + patches)
    """
    H_p = H // patch_size
    W_p = W // patch_size

    ys, xs = torch.meshgrid(
        torch.arange(H_p, device=device),
        torch.arange(W_p, device=device),
        indexing="ij"
    )

    xs = xs * patch_size + patch_size / 2
    ys = ys * patch_size + patch_size / 2

    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=-1).view(-1, 3)

    K_inv = torch.inverse(K).to(device)
    rays = pix @ K_inv.T
    rays = F.normalize(rays, dim=-1)

    cls_ray = torch.zeros(1, 3, device=device)
    return torch.cat([cls_ray, rays], dim=0)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
                 shape_meta: dict,
                 model_name: str,
                 pretrained: bool,
                 frozen: bool,
                 global_pool: str,
                 transforms: list,
                 use_group_norm: bool = False,
                 share_rgb_model: bool = False,
                 imagenet_norm: bool = False,
                 feature_aggregation: str = 'spatial_embedding',
                 downsample_ratio: int = 32,
                 position_encording: str = 'learnable'):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        rgb_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        assert global_pool == ''
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool,
            num_classes=0
        )

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False

        # ================= Feature dimension =================
        feature_dim = None
        if model_name.startswith('resnet'):
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('vit'):
            feature_dim = model.embed_dim 

        # ================= Replace BatchNorm =================
        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                    num_channels=x.num_features)
            )

        # ================= Observation shapes =================
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]

        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                key_transform_map[key] = transform
            elif type == 'low_dim' and not attr.get('ignore_by_policy', False):
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        feature_map_shape = [x // downsample_ratio for x in image_shape]

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('low_dim_keys keys:', low_dim_keys)

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.feature_aggregation = feature_aggregation
        self.feature_dim = feature_dim 

        # ================= Feature aggregation =================
        if model_name.startswith('vit'):
            if self.feature_aggregation not in (None, 'all_tokens'):
                logger.warning(f'vit will use CLS token, feature_aggregation ({self.feature_aggregation}) is ignored!')
            self.feature_aggregation = None

        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'transformer':
            if position_encording == 'learnable':
                self.position_embedding = nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim))
            elif position_encording == 'sinusoidal':
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                self.position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(2 * num_features) / feature_dim))
                self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                self.position_embedding[:, 1::2] = torch.cos(position * div_term)
            self.aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4
            )
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        # ================= Cross-view + Ray =================
        self.view_embedding = nn.Embedding(3, feature_dim)
        self.ray_proj = nn.Linear(3, feature_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            batch_first=True
        )
        self.cross_view_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.global_token = nn.Parameter(torch.randn(1, 1, feature_dim))

    def cross_view_fusion(self, token_list):
        """
        token_list: list of (B*T, N, C)
        return: (B*T, C)
        """
        fused = []

        for view_id, tokens in enumerate(token_list):
            view_emb = self.view_embedding(
                torch.tensor(view_id, device=tokens.device)
            )
            tokens = tokens + view_emb[None, None, :]
            fused.append(tokens)

        x = torch.cat(fused, dim=1)  # (B*T, sum_N, C)

        B_T = x.shape[0]
        global_token = self.global_token.expand(B_T, -1, -1)
        x = torch.cat([global_token, x], dim=1)

        x = self.cross_view_transformer(x)
        return x[:, 0]  # global token

    def aggregate_feature(self, feature):
        if self.model_name.startswith('vit'):
            assert self.feature_aggregation is None # vit uses the CLS token
            return feature[:, 0, :]
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1])
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1])
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif self.feature_aggregation == 'transformer':
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert self.feature_aggregation is None
            return feature
        
    def forward(self, obs_dict):
        features = []
        batch_size = next(iter(obs_dict.values())).shape[0]

        # ================= RGB =================
        rgb_token_list = []

        K = torch.tensor([
            [3.1066342e+04, 0.0, 1.12e+02],
            [0.0, 3.1066342e+04, 1.12e+02],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        K = K.unsqueeze(0).expand(batch_size, -1, -1)  # (B,3,3)

        for cam_id, key in enumerate(self.rgb_keys):
            img = obs_dict[key]  # (B,T,C,H,W)
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B * T, *img.shape[2:])
            img = self.key_transform_map[key](img)

            # ---------- Original feature aggregation ----------
            raw_feature = self.key_model_map[key](img)  # (B*T, feature_dim)
            feature = self.aggregate_feature(raw_feature)  
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))

            # ---------- Ray embedding ( ViT) ----------
            if self.model_name.startswith('vit'):
                H, W = self.key_shape_map[key][1:]
                assert H is not None and W is not None

                patch = self.key_model_map[key].patch_embed.patch_size[0]
                ray = compute_vit_ray_embedding(
                    H, W, patch, K[0], img.device
                )  # (num_patches, 3)
                ray = self.ray_proj(ray)
                ray = ray.unsqueeze(0).expand(raw_feature.shape[0], -1, -1)
                # feature = feature + ray[:, 0, :] 

        # ================= Low-dim =================
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))

        # ================= Concatenate all features =================
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape


if __name__=='__main__':
    shape_meta = {
        'obs': {
            'angle_rgb': {'shape': (3, 224, 224), 'type': 'rgb', 'horizon': 8},
            'top_rgb': {'shape': (3, 224, 224), 'type': 'rgb', 'horizon': 8},
            'wrist_rgb': {'shape': (3, 224, 224), 'type': 'rgb', 'horizon': 8},
            'robot0_eef_pos': {'shape': (3,), 'type': 'low_dim', 'horizon': 8},
            'robot0_eef_rot_axis_angle': {'shape': (3,), 'type': 'low_dim', 'horizon': 8},
            'robot0_eef_rot_axis_angle_wrt_start': {'shape': (3,), 'type': 'low_dim', 'horizon': 8},
            'robot0_gripper_width': {'shape': (1,), 'type': 'low_dim', 'horizon': 8},
        }
    }

    timm_obs_encoder = TimmObsEncoder(
        shape_meta=shape_meta,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        frozen=False,
        global_pool='',
        transforms=None
    )

    print("TimmObsEncoder initialized successfully")