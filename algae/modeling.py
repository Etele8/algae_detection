import types
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign, clip_boxes_to_image
from .config import CFG

# ---- SE & 6-ch input ----
class ChannelMixer1x1(torch.nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.mix = torch.nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)
        torch.nn.init.eye_(self.mix.weight.data.view(in_ch, in_ch))
        torch.nn.init.zeros_(self.mix.bias)
    def forward(self, x): return self.mix(x)

class SELayer(torch.nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(ch, ch//r, 1, bias=True), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ch//r, ch, 1, bias=True), torch.nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(self.pool(x))

def inject_se_after_layer2(backbone_body):
    layer2 = backbone_body.layer2
    orig_forward = layer2.forward
    layer2.add_module("se_after", SELayer(512, r=16))
    def new_forward(x): return layer2.se_after(orig_forward(x))
    layer2.forward = new_forward

# --- Soft-NMS (Gaussian) ---
def soft_nms_gaussian(boxes, scores, iou_thr=0.5, sigma=0.5, score_thr=1e-3, max_keep=300):
    device = boxes.device
    idxs = torch.arange(boxes.size(0), device=device)
    keep = []
    while scores.numel() > 0:
        i = torch.argmax(scores); max_idx = idxs[i]
        keep.append(int(max_idx))
        if scores.numel() == 1: break
        boxes_i = boxes[i+1:] if i>0 else boxes[1:]
        scores_i = scores[i+1:] if i>0 else scores[1:]
        idxs_i  = idxs[i+1:]  if i>0 else idxs[1:]
        from .metrics import box_iou_xyxy
        ious = box_iou_xyxy(boxes[i].unsqueeze(0), boxes_i).squeeze(0)
        decay = torch.exp(-(ious * ious) / sigma)
        scores_i = scores_i * decay
        keep_mask = scores_i >= score_thr
        boxes, scores, idxs = boxes_i[keep_mask], scores_i[keep_mask], idxs_i[keep_mask]
        if len(keep) >= max_keep: break
    return torch.tensor(keep, device=device, dtype=torch.long)

def patch_postprocess_with_softnms(model, sigma=0.5):
    def postprocess_detections_softnms(self, class_logits, box_regression, proposals, image_shapes):
        score_thresh = getattr(self, "box_score_thresh", getattr(self, "score_thresh", 0.05))
        nms_thresh   = getattr(self, "box_nms_thresh",   getattr(self, "nms_thresh",   0.5))
        dets_per_img = getattr(self, "box_detections_per_img", getattr(self, "detections_per_img", 100))

        num_classes    = class_logits.shape[-1]
        boxes_per_img  = [len(p) for p in proposals]

        pred_boxes_all  = self.box_coder.decode(box_regression, proposals)
        pred_boxes_all  = pred_boxes_all.reshape(-1, num_classes, 4).split(boxes_per_img, 0)
        pred_scores_all = torch.softmax(class_logits, -1).split(boxes_per_img, 0)

        boxes_out, scores_out, labels_out = [], [], []
        for boxes, scores, img_shape in zip(pred_boxes_all, pred_scores_all, image_shapes):
            Ni = boxes.shape[0]
            boxes = clip_boxes_to_image(boxes.reshape(-1,4), img_shape).reshape(Ni, num_classes, 4)
            boxes  = boxes[:, 1:, :]
            scores = scores[:, 1:]

            flat_boxes  = boxes.reshape(-1,4)
            flat_scores = scores.reshape(-1)
            flat_labels = torch.arange(1, num_classes, device=boxes.device).repeat(Ni)

            keep_boxes, keep_scores, keep_labels = [], [], []
            for cls_id in range(1, num_classes):
                m = (flat_labels == cls_id); b, s = flat_boxes[m], flat_scores[m]
                if b.numel() == 0: continue
                keep1 = s >= score_thresh; b, s = b[keep1], s[keep1]
                if b.numel() == 0: continue
                keep2 = soft_nms_gaussian(b, s, iou_thr=nms_thresh, sigma=sigma,
                                          score_thr=score_thresh, max_keep=dets_per_img*2)
                if keep2.numel() == 0: continue
                keep_boxes.append(b[keep2]); keep_scores.append(s[keep2])
                keep_labels.append(torch.full((keep2.numel(),), cls_id, dtype=torch.int64, device=b.device))

            if keep_boxes:
                boxes_cat  = torch.cat(keep_boxes,  0)
                scores_cat = torch.cat(keep_scores, 0)
                labels_cat = torch.cat(keep_labels, 0)
                topk = min(dets_per_img, boxes_cat.size(0))
                order = torch.argsort(scores_cat, descending=True)[:topk]
                boxes_cat, scores_cat, labels_cat = boxes_cat[order], scores_cat[order], labels_cat[order]
            else:
                dev = class_logits.device
                boxes_cat  = torch.empty((0,4), device=dev)
                scores_cat = torch.empty((0,), device=dev)
                labels_cat = torch.empty((0,), dtype=torch.int64, device=dev)

            boxes_out.append(boxes_cat); scores_out.append(scores_cat); labels_out.append(labels_cat)
        return boxes_out, scores_out, labels_out

    model.roi_heads.postprocess_detections = types.MethodType(postprocess_detections_softnms, model.roi_heads)

def get_faster_rcnn_model(num_classes=7, image_size=(1024,1024), backbone='resnet101') -> FasterRCNN:
    backbone_fpn = resnet_fpn_backbone(backbone_name=backbone, weights="DEFAULT",
                                       returned_layers=[1,2,3,4], extra_blocks=None)
    old_conv = backbone_fpn.body.conv1
    mixer = ChannelMixer1x1(in_ch=6)
    new_conv = torch.nn.Conv2d(6, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        mean_w = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:] = mean_w.repeat(1, 3, 1, 1)
    backbone_fpn.body.conv1 = torch.nn.Sequential(mixer, new_conv)
    inject_se_after_layer2(backbone_fpn.body)

    roi_pooler = MultiScaleRoIAlign(featmap_names=['0','1','2','3'], output_size=7, sampling_ratio=2)
    anchor_generator = AnchorGenerator(
        sizes=((16,24,32),(32,48,64),(64,96,128),(128,192,256),(256,348,512)),
        aspect_ratios=((0.5,1.0,2.0),)*5
    )

    model = FasterRCNN(
        backbone=backbone_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        rpn_fg_iou_thresh=0.5, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=512, rpn_positive_fraction=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.5,
        box_score_thresh=0.00, box_nms_thresh=0.5, box_detections_per_img=1000,
    )

    model.transform = GeneralizedRCNNTransform(
        min_size=[1600,1800,2048], max_size=4096, image_mean=[0.0]*6, image_std=[1.0]*6
    )
    patch_postprocess_with_softnms(model, sigma=0.5)
    return model
