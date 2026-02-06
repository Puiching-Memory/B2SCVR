#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified DAC (Domain-Adapted Codec-aware) Feature Extractor.

This module provides a unified interface that outputs:
1. Multi-scale SAM encoder features (for video completion)
2. Predicted masks (via VOS inference)

Key optimization: The SAM encoder is run only ONCE per frame, and the features
are reused for both mask prediction and video completion feature fusion.
"""

import os
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class UnifiedDACExtractor(nn.Module):
    """
    Unified DAC extractor that outputs both masks and multi-scale features.
    
    This avoids running the SAM encoder twice by caching the trunk features
    during VOS inference and returning them for use in video completion.
    
    Outputs multi-scale features in the format expected by SAMFuser:
        [T, 96, H/4, W/4], [T, 192, H/8, W/8], [T, 384, H/16, W/16], [T, 768, H/32, W/32]
    """
    
    def __init__(
        self,
        sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_checkpoint: str = None,
        device: str = "cuda",
    ):
        super().__init__()
        
        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, '/media/tianyi/BSC-VideoCompletion/B2SCVR/model/modules/sam2')
        from sam2.build_sam import build_sam2_video_predictor
        
        self.device = torch.device(device)
        
        # Default checkpoint path
        if sam2_checkpoint is None:
            sam2_checkpoint = "/media/tianyi/BSC-VideoCompletion/B2SCVR/model/modules/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_BSCV_lac_atd_dino_50eps.yaml/checkpoints/checkpoint.pt"
        
        # Build the SAM2 video predictor
        hydra_overrides_extra = ["++model.non_overlap_masks=true"]
        self.predictor = build_sam2_video_predictor(
            config_file=sam2_cfg,
            ckpt_path=sam2_checkpoint,
            apply_postprocessing=False,
            hydra_overrides_extra=hydra_overrides_extra,
            vos_optimized=False,
        )
        
        # Direct reference to the trunk for feature extraction
        self.trunk = self.predictor.image_encoder.trunk
        
        # Cache for storing features during inference
        self._feature_cache = {}
        
        # Freeze parameters (inference only)
        for param in self.parameters():
            param.requires_grad = False
    
    def extract_trunk_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from the SAM encoder trunk.
        
        Args:
            images: [B, 3, H, W] tensor normalized to [0, 1] or [-1, 1]
            
        Returns:
            List of 4 feature tensors (from low to high resolution):
                - [B, 96, H/4, W/4]   - stage 1
                - [B, 192, H/8, W/8]  - stage 2
                - [B, 384, H/16, W/16] - stage 3
                - [B, 768, H/32, W/32] - stage 4
        """
        with torch.no_grad():
            features = self.trunk(images)
        
        # The trunk returns features from all stages (low-res to high-res)
        # We need to reverse to get [high-res, ..., low-res] ordering
        # Actually, trunk outputs are already in order from early to late stages
        return features  # List of [B, C, H, W] tensors
    
    @torch.inference_mode()
    def run_vos_inference(
        self,
        video_dir: str,
        mv_dir: str,
        pm_dir: str,
        input_mask_dir: str,
        video_name: str,
        score_thresh: float = 0.0,
        use_all_masks: bool = False,
        return_trunk_features: bool = True,
    ) -> Tuple[Dict[int, np.ndarray], Optional[Dict[int, List[torch.Tensor]]], int, int]:
        """
        Run VOS inference and optionally return cached trunk features.
        
        Args:
            video_dir: Directory containing video frames (as JPEG files)
            mv_dir: Directory containing motion vector maps
            pm_dir: Directory containing prediction mode files
            input_mask_dir: Directory containing input masks
            video_name: Name of the video
            score_thresh: Threshold for mask logits
            use_all_masks: Whether to use all available masks or just first frame
            return_trunk_features: Whether to return cached trunk features
            
        Returns:
            video_segments: Dict[frame_idx, mask_array]
            trunk_features: Dict[frame_idx, List[Tensor]] or None
            height: Video height
            width: Video width
        """
        # Clear feature cache
        self._feature_cache.clear()
        
        # Initialize inference state
        video_path = os.path.join(video_dir, video_name)
        mv_path = os.path.join(mv_dir, video_name + '_2.h264')
        pm_path = os.path.join(pm_dir, video_name + '_2.h264.txt')
        
        inference_state = self.predictor.init_state(
            video_path=video_path,
            mv_path=mv_path,
            pm_path=pm_path,
            async_loading_frames=False,
        )
        
        height = inference_state["video_height"]
        width = inference_state["video_width"]
        
        # Get frame names
        frame_names = [
            os.path.splitext(p)[0]
            for p in os.listdir(video_path)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(p))  # p is already without extension
        num_frames = len(frame_names)
        
        print(f"  - Video path: {video_path}")
        print(f"  - Number of frames: {num_frames}")
        print(f"  - Frame names (first 5): {frame_names[:5]}")
        print(f"  - Looking for masks in: {os.path.join(input_mask_dir, video_name)}")
        
        # List available mask files
        mask_video_dir = os.path.join(input_mask_dir, video_name)
        if os.path.isdir(mask_video_dir):
            available_masks = [f for f in os.listdir(mask_video_dir) if f.endswith('.png')]
            print(f"  - Available mask files: {available_masks}")
        else:
            print(f"  - WARNING: Mask directory does not exist: {mask_video_dir}")
        
        # Collect input masks (per-object)
        inputs_per_object = defaultdict(dict)
        for idx, name in enumerate(frame_names):
            mask_path = os.path.join(input_mask_dir, video_name, f"{name}.png")
            if os.path.exists(mask_path):
                print(f"  - Found mask at frame {idx}: {mask_path}")
                mask = Image.open(mask_path)
                mask = np.array(mask).astype(np.uint8)
                
                print(f"    Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
                
                object_ids = np.unique(mask)
                object_ids = object_ids[object_ids > 0].tolist()
                print(f"    Object IDs found: {object_ids}")
                
                for object_id in object_ids:
                    object_mask = (mask == object_id)
                    if not np.any(object_mask):
                        continue
                    if len(inputs_per_object[object_id]) > 0 and not use_all_masks:
                        continue
                    inputs_per_object[object_id][idx] = object_mask
        
        print(f"  - Total objects collected: {list(inputs_per_object.keys())}")
        
        # Run inference for each object and collect output scores
        object_ids = sorted(inputs_per_object)
        output_scores_per_object = defaultdict(dict)
        
        for object_id in object_ids:
            input_frame_inds = sorted(inputs_per_object[object_id])
            self.predictor.reset_state(inference_state)
            
            for input_frame_idx in input_frame_inds:
                self.predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    mask=inputs_per_object[object_id][input_frame_idx],
                )
            
            # Propagate FORWARD from first mask frame
            print(f"  - Propagating object {object_id} forward from frame {min(input_frame_inds)}")
            for out_frame_idx, _, out_mask_logits in self.predictor.propagate_in_video(
                inference_state,
                start_frame_idx=min(input_frame_inds),
                reverse=False,
            ):
                obj_scores = out_mask_logits.cpu().numpy()
                output_scores_per_object[object_id][out_frame_idx] = obj_scores
                
                # Cache trunk features if requested
                if return_trunk_features and out_frame_idx not in self._feature_cache:
                    # Get the image for this frame
                    image = inference_state["images"][out_frame_idx].to(self.device).float().unsqueeze(0)
                    trunk_feats = self.extract_trunk_features(image)
                    self._feature_cache[out_frame_idx] = [f.cpu() for f in trunk_feats]
            
            # Propagate BACKWARD from first mask frame (for frames before the mask)
            if min(input_frame_inds) > 0:
                print(f"  - Propagating object {object_id} backward from frame {min(input_frame_inds)}")
                for out_frame_idx, _, out_mask_logits in self.predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=min(input_frame_inds),
                    reverse=True,
                ):
                    obj_scores = out_mask_logits.cpu().numpy()
                    output_scores_per_object[object_id][out_frame_idx] = obj_scores
        
        # Debug: Check output scores
        print(f"  - Output scores collected for objects: {list(output_scores_per_object.keys())}")
        for obj_id in output_scores_per_object:
            frames_with_scores = sorted(output_scores_per_object[obj_id].keys())
            print(f"    Object {obj_id}: frames {min(frames_with_scores)}-{max(frames_with_scores)} ({len(frames_with_scores)} frames)")
        
        # Consolidate masks (combine all objects)
        video_segments = {}
        for frame_idx in range(num_frames):
            if len(object_ids) == 0:
                video_segments[frame_idx] = np.zeros((height, width), dtype=np.uint8)
                continue
            
            scores = torch.full(
                size=(len(object_ids), 1, height, width),
                fill_value=-1024.0,
                dtype=torch.float32,
            )
            for i, obj_id in enumerate(object_ids):
                if frame_idx in output_scores_per_object[obj_id]:
                    scores[i] = torch.from_numpy(output_scores_per_object[obj_id][frame_idx])
            
            # Apply non-overlapping constraints
            scores = self.predictor._apply_non_overlapping_constraints(scores)
            
            # Combine into single mask (use 1 as object value, not obj_id which can be 255)
            mask = np.zeros((height, width), dtype=np.uint8)
            for i, obj_id in enumerate(sorted(object_ids)[::-1]):
                obj_mask = (scores[i] > score_thresh).cpu().numpy().squeeze()
                mask[obj_mask] = 1  # Use 1 instead of obj_id (255)
            
            video_segments[frame_idx] = mask
        
        # Debug: Check consolidated masks
        non_zero_segments = sum(1 for m in video_segments.values() if np.sum(m) > 0)
        print(f"  - Consolidated masks with non-zero pixels: {non_zero_segments}/{len(video_segments)}")
        
        # Return results
        trunk_features = dict(self._feature_cache) if return_trunk_features else None
        self._feature_cache.clear()
        
        return video_segments, trunk_features, height, width
    
    def get_features_for_frames(
        self,
        frames: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> List[torch.Tensor]:
        """
        Extract multi-scale features for a batch of frames.
        
        This is a direct feature extraction method (without VOS inference).
        
        Args:
            frames: [T, 3, H, W] tensor of frames (values in [0, 255] or normalized)
            target_size: Optional (H, W) to resize features to
            
        Returns:
            List of 4 feature tensors, each of shape [T, C, H', W']
        """
        # Ensure frames are on device and properly normalized
        if frames.max() > 1.0:
            frames = frames / 255.0
        
        frames = frames.to(self.device)
        
        with torch.no_grad():
            features = self.trunk(frames)
        
        return features


class DACFeatureExtractorForCompletion(nn.Module):
    """
    A drop-in replacement for SAM2Extractor in video completion models.
    
    This class can be used directly in place of feat_sam_extract.SAM2Extractor,
    but it can also receive pre-computed features from UnifiedDACExtractor
    to avoid redundant computation.
    
    Usage in bscvr_hq_p2_sam_v3.py:
        # Replace:
        # self.sam2_encoder = SAM2Extractor()
        # With:
        self.sam2_encoder = DACFeatureExtractorForCompletion()
        
        # Then in forward(), you can optionally pass pre-computed features:
        # sam2_feat = self.sam2_encoder(corrupted_frames_, precomputed_features=cached_feats)
    """
    
    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = "cuda",
    ):
        super().__init__()
        
        import sys
        sys.path.insert(0, '/media/tianyi/BSC-VideoCompletion/B2SCVR/model/modules/sam2')
        from sam2.build_sam import build_sam2
        
        if checkpoint_path is None:
            checkpoint_path = "/media/tianyi/BSC-VideoCompletion/B2SCVR/model/modules/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_BSCV_lac_atd_dino_50eps.yaml/checkpoints/checkpoint.pt"
        
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        model = build_sam2(model_cfg, checkpoint_path)
        
        # Only keep the trunk (encoder)
        self.encoder = model.image_encoder.trunk
        
        # Clean up unused components
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        
        # Freeze parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        x: torch.Tensor,
        precomputed_features: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Extract or return multi-scale features.
        
        Args:
            x: [B, 3, H, W] input tensor
            precomputed_features: Optional pre-computed features from UnifiedDACExtractor
            
        Returns:
            List of 4 feature tensors:
                [B, 96, H/4, W/4], [B, 192, H/8, W/8], 
                [B, 384, H/16, W/16], [B, 768, H/32, W/32]
        """
        if precomputed_features is not None:
            # Use pre-computed features (from DAC VOS inference)
            # May need to resize to match input size
            return self._adapt_features(precomputed_features, x.shape[-2:])
        
        # Compute features from scratch
        with torch.no_grad():
            sam_feat = self.encoder(x)
        return sam_feat
    
    def _adapt_features(
        self,
        features: List[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> List[torch.Tensor]:
        """
        Adapt pre-computed features to match expected output sizes.
        """
        # Expected output sizes based on input size (H, W)
        H, W = target_size
        expected_sizes = [
            (H // 4, W // 4),
            (H // 8, W // 8),
            (H // 16, W // 16),
            (H // 32, W // 32),
        ]
        
        adapted = []
        for feat, (eh, ew) in zip(features, expected_sizes):
            if feat.shape[-2:] != (eh, ew):
                feat = F.interpolate(
                    feat, size=(eh, ew), mode='bilinear', align_corners=False
                )
            adapted.append(feat)
        
        return adapted


# ============================================================================
# Convenience function for creating the unified extractor
# ============================================================================

def create_unified_dac_extractor(
    checkpoint_path: str = None,
    device: str = "cuda",
) -> UnifiedDACExtractor:
    """
    Create a UnifiedDACExtractor with default settings.
    
    Args:
        checkpoint_path: Path to DAC checkpoint (uses default if None)
        device: Device to use
        
    Returns:
        UnifiedDACExtractor instance
    """
    if checkpoint_path is None:
        checkpoint_path = "/media/tianyi/BSC-VideoCompletion/B2SCVR/model/modules/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_BSCV_lac_atd_dino_50eps.yaml/checkpoints/checkpoint.pt"
    
    return UnifiedDACExtractor(
        sam2_checkpoint=checkpoint_path,
        device=device,
    )
