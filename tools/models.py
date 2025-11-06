from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

class ChannelMixer1x1(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.mix = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.mix.bias)
            self.mix.weight.zero_()
            eye = torch.eye(in_ch)[:, :, None, None]
            self.mix.weight.copy_(eye)
    def forward(self, x): return self.mix(x)

class SELayer(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

def inject_se_after_layer2(backbone_body: nn.Module):
    layer2 = backbone_body.layer2
    orig_forward = layer2.forward
    layer2.add_module("se_after", SELayer(512, r=16))
    def new_forward(x):
        y = orig_forward(x)
        return layer2.se_after(y)
    layer2.forward = new_forward

# --- exact re-build of the old detector (torchvision 0.20.1) ---
def build_old_frcnn_6ch_se(num_classes=7) -> FasterRCNN:
    # 1) ResNet101-FPN; IMPORTANT: ask for 5 levels via LastLevelMaxPool
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights="DEFAULT",                               # don't load the 3ch FPN weights here
        returned_layers=[1,2,3,4],
        extra_blocks=LastLevelMaxPool()
    )

    # 2) Replace conv1 by [ChannelMixer1x1(6) -> Conv2d(6->64,7x7)]
    old_conv = backbone.body.conv1
    mix = ChannelMixer1x1(in_ch=6)
    new_conv = nn.Conv2d(6, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        # copy RGB weights to first 3 channels; average RGB into the extra 3
        new_conv.weight[:, :3] = old_conv.weight
        mean_w = old_conv.weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
        new_conv.weight[:, 3:] = mean_w.repeat(1, 3, 1, 1)
    backbone.body.conv1 = nn.Sequential(mix, new_conv)   # keys like conv1.0.mix.*, conv1.1.*

    # 3) Inject SE after layer2
    inject_se_after_layer2(backbone.body)

    # 4) RPN anchors → 9 anchors/location on each of the 5 FPN maps
    #    (these sizes/ratios match your checkpoint heads: cls_logits out_channels=9, bbox_pred=36)
    sizes = (
        (16, 24, 32),     # P2
        (32, 48, 64),     # P3
        (64, 96, 128),    # P4
        (128, 192, 256),  # P5
        (256, 384, 512),  # P6 (LastLevelMaxPool)
    )
    ratios = ((0.5, 1.0, 2.0),) * len(sizes)
    anchor_gen = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)

    # 5) ROI pool over the 5 maps (note 'pool' is the max-pooled level in torchvision)
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0','1','2','3','pool'], output_size=7, sampling_ratio=2)

    # 6) Build detector
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_roi_pool=roi_pooler,
        # sampler
        rpn_fg_iou_thresh=0.5, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=512, rpn_positive_fraction=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.5,
        # postproc defaults (you can override at inference)
        box_score_thresh=0.0, box_nms_thresh=0.5, box_detections_per_img=1000
    )

    # 7) 6-channel transform (matches your fused cache)
    model.transform = GeneralizedRCNNTransform(
        min_size=[1600, 1800, 2048, 2300],  # what you had
        max_size=4096,
        image_mean=[0.0]*6,
        image_std=[1.0]*6,
    )
    return model

# --- loader that prints a quick sanity readout and tolerates key name diffs ---
def load_single_model(ckpt_path: str, device, warp_S: int) -> torch.nn.Module:
    sd = torch.load(ckpt_path, map_location=device)
    # handle {"model_state":...} or raw state dict
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]

    model = build_old_frcnn_6ch_se(num_classes=7).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    # quick assert: 9 anchors/location?
    A = model.rpn.head.cls_logits.out_channels
    assert A == 9, f"RPN anchors/location = {A}, expected 9 (3 sizes × 3 ratios)"
    B = model.rpn.head.bbox_pred.out_channels
    assert B == 36, f"RPN bbox channels = {B}, expected 36 (4 × 9)"

    model.eval()
    model.transform.min_size = [warp_S]
    model.transform.max_size = warp_S
    model.roi_heads.score_thresh = 0.0
    model.roi_heads.nms_thresh   = 0.85
    model.roi_heads.detections_per_img = 20000
    if hasattr(model.rpn, "_pre_nms_top_n"):
        model.rpn._pre_nms_top_n  = {"training": 20000, "testing": 20000}
        model.rpn._post_nms_top_n = {"training": 10000, "testing": 10000}
    model.rpn.nms_thresh = 0.85
    return model

# ======================COLONY==========================

def patch_roi_heads_to_ignore_crowd(roi_heads, iou_thr=0.5):
    """
    Monkey-patch RoIHeads so proposals overlapping crowd_boxes are ignored during training.
    """
    orig_assign = roi_heads.assign_targets_to_proposals

class ChannelMixer1x1(torch.nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.mix = torch.nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)
        torch.nn.init.eye_(self.mix.weight.data.view(in_ch, in_ch))
        torch.nn.init.zeros_(self.mix.bias)
    def forward(self, x):
        return self.mix(x)

class SELayer(torch.nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(ch, ch // r, 1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ch // r, ch, 1, bias=True),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

def inject_se_after_layer2(backbone_body):
    layer2 = backbone_body.layer2
    orig_forward = layer2.forward
    layer2.add_module("se_after", SELayer(512, r=16))
    def new_forward(x):
        y = orig_forward(x)
        return layer2.se_after(y)
    layer2.forward = new_forward


def get_colony_model(num_classes=7, backbone="resnet101") -> FasterRCNN:

    backbone_fpn = resnet_fpn_backbone(
        backbone_name=backbone, weights="DEFAULT",
        returned_layers=[1, 2, 3, 4], extra_blocks=None,
    )

    old_conv = backbone_fpn.body.conv1
    mixer = ChannelMixer1x1(in_ch=6)
    new_conv = torch.nn.Conv2d(6, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        mean_w = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:] = mean_w.repeat(1, 3, 1, 1)
    backbone_fpn.body.conv1 = torch.nn.Sequential(mixer, new_conv)
    inject_se_after_layer2(backbone_fpn.body)
    
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7, sampling_ratio=2
    )
    anchor_generator = AnchorGenerator(
        sizes = (
            (8, 12, 18),      # finer detail tier
            (24, 32, 48),
            (72, 96, 128),
            (160, 224, 300),
            (450, 600, 800),
        ),
        aspect_ratios=((0.8, 1.0, 1.4),) * 5
    )
    
    model = FasterRCNN(
        backbone=backbone_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        rpn_fg_iou_thresh=0.45,
        rpn_bg_iou_thresh=0.25,
        rpn_batch_size_per_image=512,
        rpn_positive_fraction=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.5,
        box_score_thresh=0.00,
        box_nms_thresh=0.5,
        box_detections_per_img=1200,
    )

    patch_roi_heads_to_ignore_crowd(model.roi_heads, iou_thr=0.5)
    
    model.transform = GeneralizedRCNNTransform(
        min_size=[1600, 1800, 2048],   # multi-scale
        max_size=4096,
        image_mean=[0.0] * 6,
        image_std=[1.0] * 6,
    )

    return model

def load_colony_model(ckpt_path: Path, device, min_size: int) -> torch.nn.Module:
    model = get_colony_model(num_classes=6, backbone="resnet101").to(device)
    d = torch.load(str(ckpt_path), map_location="cpu")
    state = d.get("model_state", d) if isinstance(d, dict) else d
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Load] {ckpt_path.name}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    if hasattr(model.roi_heads, "score_thresh"):
        model.roi_heads.score_thresh = 0.0
    if hasattr(model, "transform"):
        model.transform.min_size = [int(min_size)]
        
    model.to(device)
    if hasattr(model.rpn, "_pre_nms_top_n"):
        model.rpn._pre_nms_top_n  = {"training": 20000, "testing": 20000}
        model.rpn._post_nms_top_n = {"training": 10000, "testing": 10000}
    model.rpn.nms_thresh = 0.85
    model.roi_heads.score_thresh = 0.0
    model.roi_heads.nms_thresh   = 0.85
    model.roi_heads.detections_per_img = 20000
    
    return model