"""
YOLOX Object Detection Implementation
Optimized for accurate object detection with proper coordinate handling
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import json
import os
import warnings
import torch.nn.functional as F
from typing import List, Dict, Tuple
from collections import Counter
from model_tests import ConfusionMatrix
from COCO_CLASSES import INDOOR_BUSINESS_CLASSES

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Apple Silicon optimizations
if torch.backends.mps.is_available():
    print("Apple detected - enabling MPS optimizations")
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'


class SiLU(nn.Module):
    """Sigmoid Linear Unit activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, 
                             padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = self.get_activation(act, inplace=True)

    def get_activation(self, act="silu", inplace=True):
        if act == "silu":
            return SiLU()
        elif act == "relu":
            return nn.ReLU(inplace=inplace)
        elif act == "lrelu":
            return nn.LeakyReLU(0.1, inplace=inplace)
        else:
            return nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    """Focus operation for YOLOX"""
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)


class Bottleneck(nn.Module):
    """Standard bottleneck with shortcut connection"""
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    """CSP Bottleneck with 3 convolutions"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPDarknet(nn.Module):
    """CSP-Darknet backbone for YOLOX"""
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu"):
        super().__init__()
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        # Stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # Stages
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class YOLOPAFPN(nn.Module):
    """YOLO PANet Feature Pyramid Network"""
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), 
                 in_channels=[256, 512, 1024], depthwise=False, act="silu"):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, out_features=in_features, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[1] * width), 
                             round(3 * depth), False, depthwise=depthwise, act=act)

        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[0] * width), 
                             round(3 * depth), False, depthwise=depthwise, act=act)

        # Bottom-up conv
        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[1] * width), 
                             round(3 * depth), False, depthwise=depthwise, act=act)

        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[2] * width), 
                             round(3 * depth), False, depthwise=depthwise, act=act)

    def forward(self, input):
        """Forward pass through backbone and FPN"""
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], 1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, x2], 1)
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


class YOLOXHead(nn.Module):
    """YOLOX Detection Head"""
    def __init__(self, num_classes=80, width=1.0, strides=[8, 16, 32], 
                 in_channels=[256, 512, 1024], act="silu", depthwise=False):
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(strides)):
            self.stems.append(BaseConv(int(in_channels[i] * width), int(256 * width), 1, 1, act=act))
            
            self.cls_convs.append(nn.Sequential(*[
                Conv(int(256 * width), int(256 * width), 3, 1, act=act), 
                Conv(int(256 * width), int(256 * width), 3, 1, act=act)
            ]))
            
            self.reg_convs.append(nn.Sequential(*[
                Conv(int(256 * width), int(256 * width), 3, 1, act=act), 
                Conv(int(256 * width), int(256 * width), 3, 1, act=act)
            ]))
            
            self.cls_preds.append(nn.Conv2d(int(256 * width), self.n_anchors * self.num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(int(256 * width), 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(int(256 * width), self.n_anchors * 1, 1, 1, 0))

        self.strides = strides
        self.grids = [torch.zeros(1)] * len(strides)

    def forward(self, xin, labels=None, imgs=None):
        """Forward pass for YOLOX head"""
        outputs = []
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_feat = cls_conv(x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if not self.training:
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            outputs.append(output)

        if not self.training:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def decode_outputs(self, outputs, dtype):
        """Decode model outputs to get bounding boxes"""
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs


class YOLOX(nn.Module):
    """Complete YOLOX model"""
    def __init__(self, num_classes=80, depth=1.33, width=1.25):
        super().__init__()
        self.num_classes = num_classes
        backbone = YOLOPAFPN(depth, width)
        head = YOLOXHead(num_classes, width)
        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs, targets, x)
        return outputs


class YOLOXDetector:
    """YOLOX Object Detection System"""
    
    def __init__(self, model_path: str = "yolox_x.pth", device: str = "auto"):
        self.device = self._get_device(device)
        self.model = None
        self.input_size = (640, 640)
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.45
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
        
        self._load_model(model_path)
    
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device"""
        if device == "auto":
            # Use CPU for YOLOX due to MPS compatibility issues
            print("Using CPU for YOLOX (MPS compatibility)")
            return torch.device("cpu")
        return torch.device(device)
    
    def _load_model(self, model_path: str):
        """Load YOLOX model"""
        if not os.path.exists(model_path):
            print(f"[ERROR] YOLOX model file not found: {model_path}")
            return
        
        try:
            print(f"Loading YOLOX from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.model = YOLOX(num_classes=80, depth=1.33, width=1.25)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Clean state dict keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            print("[OK] YOLOX loaded successfully!")
            
        except Exception as e:
            print(f"[ERROR] Failed to load YOLOX: {e}")
            self.model = None
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, float]:
        """Preprocess image for YOLOX (expects 0-255 range)"""
        height, width = image.shape[:2]
        r = min(self.input_size[0] / width, self.input_size[1] / height)
        new_width, new_height = int(width * r), int(height * r)
        
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Pad image to input size
        padded_image = np.ones((self.input_size[1], self.input_size[0], 3), dtype=np.uint8) * 114
        padded_image[:new_height, :new_width] = resized_image
        
        # Convert to tensor WITHOUT normalization (YOLOX expects 0-255 range)
        image_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).float()
        return image_tensor.unsqueeze(0).to(self.device), r
    
    def postprocess_detections(self, outputs: torch.Tensor, original_shape: Tuple[int, int], ratio: float) -> List[Dict]:
        """Post-process YOLOX outputs"""
        detections: List[Dict] = []
        
        if outputs is None:
            return detections
        
        # Apply confidence threshold
        conf_mask = outputs[..., 4] > self.confidence_threshold
        outputs = outputs[conf_mask]
        
        if outputs.shape[0] == 0:
            return detections
        
        # Get class scores and predictions
        class_conf = outputs[:, 5:].max(1, keepdim=True)[0]
        class_pred = outputs[:, 5:].argmax(1, keepdim=True)
        
        # Filter by confidence
        conf_mask = (outputs[:, 4] * class_conf.squeeze()) >= self.confidence_threshold
        outputs = outputs[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        
        if outputs.shape[0] == 0:
            return detections
        
        # Convert center format to corner format
        box_corner = outputs.new(outputs.shape)
        box_corner[:, 0] = outputs[:, 0] - outputs[:, 2] / 2
        box_corner[:, 1] = outputs[:, 1] - outputs[:, 3] / 2
        box_corner[:, 2] = outputs[:, 0] + outputs[:, 2] / 2
        box_corner[:, 3] = outputs[:, 1] + outputs[:, 3] / 2
        outputs[:, :4] = box_corner[:, :4]
        
        # Apply NMS
        try:
            from torchvision.ops import nms
            nms_out_index = nms(
                outputs[:, :4],
                outputs[:, 4] * class_conf.squeeze(),
                self.nms_threshold,
            )
        except ImportError:
            # Fallback: use all detections if NMS is not available
            print("[WARNING]  NMS not available, using all detections")
            nms_out_index = torch.arange(outputs.shape[0])
        
        outputs = outputs[nms_out_index]
        class_conf = class_conf[nms_out_index]
        class_pred = class_pred[nms_out_index]
        
        # Scale back to original image size
        original_h, original_w = original_shape
        for i in range(outputs.shape[0]):
            x1 = (outputs[i, 0] / ratio).item()
            y1 = (outputs[i, 1] / ratio).item()
            x2 = (outputs[i, 2] / ratio).item()
            y2 = (outputs[i, 3] / ratio).item()
            
            # Clamp to image bounds
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))
            
            confidence = (outputs[i, 4] * class_conf[i]).item()
            class_id = class_pred[i].item()
            
            if confidence >= self.confidence_threshold and class_id < len(self.class_names):
                detection = {
                    "object": self.class_names[class_id],
                    "confidence": float(confidence),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "detection_type": "yolox"
                }
                detections.append(detection)
        
        return detections
    
    def detect_objects(self, image_path: str) -> List[Dict]:
        """Detect objects in an image using YOLOX"""
        if self.model is None:
            print("[ERROR] YOLOX model not loaded")
            return []
        
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return []
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return []
        
        original_shape = image.shape[:2]
        
        try:
            # Create matrix from JSON ground truth
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            ground_truth_file = f"test_photos/labels/{base_name}_test.json"

            # Fallback to regular json file if _test.json doesn't exist
            if not os.path.exists(ground_truth_file):
                ground_truth_file = f"test_photos/labels/{base_name}.json"

            # Final fallback to sample_test.json in current directory
            if not os.path.exists(ground_truth_file):
                ground_truth_file = "sample_test.json"

            if os.path.exists(ground_truth_file):
                with open(ground_truth_file, 'r') as f:
                    ground_truth = json.load(f)
                matrix = ConfusionMatrix(ground_truth)
                print(f" Created confusion matrix from {ground_truth_file}")
            else:
                print(f"[WARNING] Ground truth file not found: {ground_truth_file}")
                matrix = None

            print(f" Running YOLOX on {image_path}")
            input_tensor, ratio = self.preprocess_image(image)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                detections = self.postprocess_detections(outputs, original_shape, ratio)
                print(f"[OK] YOLOX detected {len(detections)} objects")

                # Handle each detection through confusion matrix
                if matrix is not None:
                    for detection in detections:
                        detected_class = detection["object"]
                        bbox = detection["bbox"]
                        matrix.handle_object_data(detected_class, bbox)

                # Get matrix metrics
                if matrix is not None:
                    class_metrics, mean_ap, mean_f1 = matrix.get_matrix_metrics()
                    print(f" Confusion Matrix Results: mAP={mean_ap:.3f}, mF1={mean_f1:.3f}")

                    # Print per-class metrics for detected classes
                    detected_classes = set(d["object"] for d in detections)
                    for metric in class_metrics:
                        if metric.class_name in detected_classes:
                            print(f"  {metric.class_name}: P={metric.precision:.3f}, R={metric.sensitivity:.3f}, F1={metric.f1_score:.3f}")

                # Save comparison visualization image (shows both detections and ground truth)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                vis_path = f"outputs/yolox_{base_name}_comparison.jpg"
                self.create_comparison_visualization(image_path, detections, matrix, vis_path)

                return detections

        except Exception as e:
            print(f"[ERROR] YOLOX failed: {e}")
            return []
    
    def create_visualization(self, image_path: str, detections: List[Dict], save_path: str = None):
        """Create visualization of YOLOX detections"""
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # YOLOX-specific color scheme
        colors = {
            "person": (255, 0, 0),         # Red
            "chair": (0, 255, 0),          # Green
            "dining table": (255, 0, 0),   # Red
            "couch": (0, 255, 0),          # Green
            "bottle": (255, 165, 0),       # Orange
            "cup": (255, 255, 0),          # Yellow
            "laptop": (0, 0, 255),         # Blue
            "mouse": (0, 0, 255),          # Blue
            "keyboard": (0, 0, 255),       # Blue
            "tv": (0, 0, 255),             # Blue
            "book": (255, 0, 255),         # Magenta
            "clock": (128, 0, 128),        # Purple
            "potted plant": (0, 255, 255), # Cyan
        }
        
        # Draw detections
        for det in detections:
            bbox = det["bbox"]
            obj = det["object"]
            conf = det["confidence"]
            
            # Choose color (default gray if not in color scheme)
            color = colors.get(obj, (128, 128, 128))
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box (thicker for YOLOX)
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with YOLOX prefix
            label = f" YOLOX: {obj} ({conf:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Position label
            text_x = x1
            text_y = y1 - 10
            if text_y < 20:
                text_y = y1 + 25
            
            # Text background
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(image_rgb, (text_x, text_y - text_height - 5),
                         (text_x + text_width, text_y + 5), color, -1)
            
            # White text
            cv2.putText(image_rgb, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image_bgr)
            print(f" YOLOX visualization saved: {save_path}")
        
        return image_rgb

    def create_comparison_visualization(self, image_path: str, detections: List[Dict], matrix: ConfusionMatrix, save_path: str = None):
        """Create visualization showing both model detections and ground truth annotations"""
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return None

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Colors for different types of boxes
        colors = {
            "detection": (0, 255, 0),      # Green for model detections
            "ground_truth": (255, 0, 0),   # Red for ground truth
            "missing": (0, 0, 255),       # Blue for missing ground truth
            "false_positive": (255, 255, 0)  # Yellow for false positives
        }

        # Draw model detections (green boxes)
        for det in detections:
            bbox = det["bbox"]
            obj = det.get("object", det.get("class", "unknown"))  # Handle both 'object' and 'class' keys
            conf = det["confidence"]

            x1, y1, x2, y2 = bbox

            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), colors["detection"], 2)

            # Draw label
            label = f" {obj}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            # Position label
            text_x = x1
            text_y = y1 - 5
            if text_y < 15:
                text_y = y1 + 20

            # Text background
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(image_rgb, (text_x, text_y - text_height - 3),
                         (text_x + text_width, text_y + 3), colors["detection"], -1)

            # White text
            cv2.putText(image_rgb, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Draw ground truth annotations (red boxes)
        ground_truth_objects = matrix.reference_object_array
        for gt_obj in ground_truth_objects:
            bbox = gt_obj.bbox
            class_name = gt_obj.class_name

            x1, y1, x2, y2 = bbox

            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), colors["ground_truth"], 2)

            # Draw label
            label = f" GT: {class_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            # Position label
            text_x = x1
            text_y = y1 - 5
            if text_y < 15:
                text_y = y1 + 20

            # Text background
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(image_rgb, (text_x, text_y - text_height - 3),
                         (text_x + text_width, text_y + 3), colors["ground_truth"], -1)

            # White text
            cv2.putText(image_rgb, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Draw missing ground truth objects (blue boxes with dashed lines)
        missing_objects = matrix.missing_objects
        for missing_obj in missing_objects:
            bbox = missing_obj.bbox
            class_name = missing_obj.class_name

            x1, y1, x2, y2 = bbox

            # Draw bounding box with dashed line
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), colors["missing"], 2, cv2.LINE_4)

            # Draw label
            label = f"[ERROR] MISSING: {class_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            # Position label
            text_x = x1
            text_y = y1 - 5
            if text_y < 15:
                text_y = y1 + 20

            # Text background
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(image_rgb, (text_x, text_y - text_height - 3),
                         (text_x + text_width, text_y + 3), colors["missing"], -1)

            # White text
            cv2.putText(image_rgb, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Add legend
        legend_y = 20
        legend_items = [
            (" Model Detection", colors["detection"]),
            (" Ground Truth", colors["ground_truth"]),
            ("[ERROR] Missing GT", colors["missing"])
        ]

        for text, color in legend_items:
            cv2.putText(image_rgb, text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            legend_y += 25

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image_bgr)
            print(f" Comparison visualization saved: {save_path}")

        return image_rgb

    def analyze_detections(self, detections: List[Dict]) -> Dict:
        """Analyze YOLOX detections"""
        objects = [det["object"] for det in detections]
        
        # COCO categories relevant to office/indoor scenes
        categories = {
            "people": ["person"],
            "furniture": ["chair", "couch", "bed", "dining table", "toilet"],
            "technology": ["laptop", "mouse", "remote", "keyboard", "cell phone", "tv"],
            "kitchen": ["microwave", "oven", "toaster", "sink", "refrigerator"],
            "supplies": ["book", "scissors"],
            "containers": ["bottle", "wine glass", "cup", "bowl", "vase"],
            "personal": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
            "misc": ["clock", "potted plant", "teddy bear", "hair drier", "toothbrush"]
        }
        
        # Count objects by category
        category_counts = {}
        for category, keywords in categories.items():
            count = sum(1 for obj in objects if obj in keywords)
            if count > 0:
                category_counts[category] = count
        
        # Get object counts
        object_counts = Counter(objects)
        
        return {
            "total_objects": len(detections),
            "categories": category_counts,
            "object_counts": dict(object_counts.most_common()),
            "unique_objects": len(set(objects)),
            "confidence_stats": {
                "min": min([det["confidence"] for det in detections]) if detections else 0,
                "max": max([det["confidence"] for det in detections]) if detections else 0,
                "avg": sum([det["confidence"] for det in detections]) / len(detections) if detections else 0
            }
        }
    
    def save_results(self, image_path: str, detections: List[Dict], output_dir: str = "outputs"):
        """Save YOLOX detection results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save JSON results
        json_path = f"{output_dir}/yolox_{base_name}_detections.json"
        analysis = self.analyze_detections(detections)
        
        result_data = {
            "model": "YOLOX",
            "image": image_path,
            "detections": detections,
            "analysis": analysis,
            "model_info": {
                "confidence_threshold": self.confidence_threshold,
                "nms_threshold": self.nms_threshold,
                "input_size": self.input_size
            }
        }
        
        with open(json_path, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f" YOLOX results saved: {json_path}")
        
        # Save visualization
        vis_path = f"{output_dir}/yolox_{base_name}_visualization.jpg"
        self.create_visualization(image_path, detections, vis_path)
        
        return json_path, vis_path


def yolox(image_path, reference_json_path, use_gpu=True, create_overlay=True):
    """
    Main YOLOX function for object detection and confusion matrix analysis
    
    Args:
        image_path: Path to the image file to analyze
        reference_json_path: Path to ground truth JSON file
        use_gpu: Whether to use GPU if available (default: True)
        create_overlay: Whether to create overlay visualization (default: True)
    
    Returns:
        dict: Formatted confusion matrix results (includes 'overlay_path' if created)
    """
    print("Running YOLOX model...")
    
    # Load reference data
    with open(reference_json_path, 'r') as f:
        reference_data = json.load(f)
    
    # Handle different JSON formats
    # Format 1: [{"class": "...", "bbox": [...]}, ...]
    # Format 2: {"image": "...", "objects": [{"class": "...", "bbox": [...]}]}
    if isinstance(reference_data, dict) and 'objects' in reference_data:
        reference_data = reference_data['objects']
    
    # Detect objects in the image
    detected_objects = detect_objects_in_image(image_path, use_gpu)
    
    # Create confusion matrix and process detections
    confusion_matrix = ConfusionMatrix(reference_data)
    
    # Process each detected object through the confusion matrix
    for obj in detected_objects:
        confusion_matrix.handle_object_data(obj['class'], obj['bbox'])
    
    # Get metrics and format results
    class_metrics, mean_ap, mean_f1 = confusion_matrix.get_matrix_metrics()
    
    # Format and print results
    results = format_results(class_metrics, mean_ap, mean_f1, confusion_matrix)
    print_results(results)
    
    return results


def detect_objects_in_image(image_path, use_gpu=True):
    """
    Detect objects in an image using YOLOX model
    
    Args:
        image_path: Path to the image file
        use_gpu: Whether to use GPU if available
    
    Returns:
        list: List of detected objects with format [{'class': str, 'bbox': [x1,y1,x2,y2]}, ...]
    """
    try:
        # Create detector
        detector = YOLOXDetector()
        
        if detector.model is None:
            print("[ERROR] YOLOX model not loaded")
            return []
        
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return []
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return []
        
        original_shape = image.shape[:2]
        
        print(f" Running YOLOX on {image_path}")
        input_tensor, ratio = detector.preprocess_image(image)
        
        with torch.no_grad():
            outputs = detector.model(input_tensor)
            detections = detector.postprocess_detections(outputs, original_shape, ratio)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Convert to format expected by model_tests
        detected_objects = []
        for det in detections:
            detected_objects.append({
                'class': det['object'],
                'bbox': det['bbox'],
                'confidence': det['confidence']
            })
        
        print(f"Detected {len(detected_objects)} relevant objects")
        return detected_objects
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return []


def get_coco_class_mapping():
    """
    Create mapping from COCO class indices to indoor business class names
    
    Returns:
        dict: Mapping from COCO class name to indoor class name
    """
    # Map COCO classes to indoor business classes
    # YOLOX already outputs COCO class names, so this is mostly a passthrough
    # that filters to only indoor business classes
    mapping = {}
    for class_name in INDOOR_BUSINESS_CLASSES:
        mapping[class_name] = class_name
    return mapping


def format_results(class_metrics, mean_ap, mean_f1, confusion_matrix):
    """
    Format confusion matrix results for display
    
    Args:
        class_metrics: List of ObjectMetrics objects
        mean_ap: Mean average precision
        mean_f1: Mean F1 score
        confusion_matrix: ConfusionMatrix object
    
    Returns:
        dict: Formatted results
    """
    results = {
        'mean_average_precision': mean_ap,
        'mean_f1_score': mean_f1,
        'class_metrics': {},
        'confusion_matrix': confusion_matrix.get_confusion_matrix().tolist(),
        'unmatched_objects': len(confusion_matrix.unmatched_objects),
        'missing_objects': len(confusion_matrix.missing_objects)
    }
    
    # Format class-specific metrics
    for metric in class_metrics:
        results['class_metrics'][metric.class_name] = {
            'precision': metric.precision,
            'sensitivity': metric.sensitivity,
            'specificity': metric.specificity,
            'f1_score': metric.f1_score
        }
    
    return results


def print_results(results):
    """
    Print formatted confusion matrix results
    
    Args:
        results: Formatted results dictionary
    """
    print("\n" + "="*60)
    print("YOLOX CONFUSION MATRIX RESULTS")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Mean Average Precision: {results['mean_average_precision']:.4f}")
    print(f"  Mean F1 Score: {results['mean_f1_score']:.4f}")
    print(f"  Unmatched Objects: {results['unmatched_objects']}")
    print(f"  Missing Objects: {results['missing_objects']}")
    
    print(f"\nClass-Specific Metrics:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1 Score':<10}")
    print("-" * 60)
    
    for class_name, metrics in results['class_metrics'].items():
        print(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['sensitivity']:<12.4f} "
              f"{metrics['specificity']:<12.4f} {metrics['f1_score']:<10.4f}")
    
    print("\n" + "="*60)


def yolox_only(image1_path: str = "IMG_1464.jpg", image2_path: str = "IMG_1465.jpg"):
    """YOLOX-only detection for timing comparison"""
    print("Running YOLOX-only detection...")
    
    detector = YOLOXDetector()
    if detector.model is None:
        return "YOLOX not available"
    
    total_objects = 0
    for img_path in [image1_path, image2_path]:
        if os.path.exists(img_path):
            detections = detector.detect_objects(img_path)
            total_objects += len(detections)
    
    return f"YOLOX completed - {total_objects} objects detected"


def comprehensive_yolox_analysis(image1_path: str = "IMG_1464.jpg", image2_path: str = "IMG_1465.jpg"):
    """Comprehensive YOLOX-only analysis with file outputs"""
    print("=" * 60)
    print(" YOLOX COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    detector = YOLOXDetector()
    
    if detector.model is None:
        print("[ERROR] Cannot run - YOLOX not available")
        return
    
    # Process images
    all_results = {}
    total_objects = 0
    
    for img_path in [image1_path, image2_path]:
        if not os.path.exists(img_path):
            print(f"[ERROR] {img_path} not found")
            continue
        
        print(f"\n Processing {img_path}")
        print("-" * 40)
        
        # Run detection
        detections = detector.detect_objects(img_path)
        
        if detections:
            # Save results with YOLOX-specific naming
            json_path, vis_path = detector.save_results(img_path, detections)
            
            # Analyze results
            analysis = detector.analyze_detections(detections)
            
            print(f"[OK] Objects detected: {len(detections)}")
            print(f" Results saved: {json_path}")
            print(f"  Visualization: {vis_path}")
            print(f" Categories: {analysis['categories']}")
            print(f" Confidence range: {analysis['confidence_stats']['min']:.3f} - {analysis['confidence_stats']['max']:.3f}")
            
            # Show top detections
            if len(detections) > 0:
                print(" Top detections:")
                sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
                for i, det in enumerate(sorted_detections[:5], 1):
                    print(f"  {i}. {det['object']}: {det['confidence']:.3f}")
            
            all_results[img_path] = {
                "detections": detections,
                "analysis": analysis,
                "files": {"json": json_path, "visualization": vis_path}
            }
            total_objects += len(detections)
        else:
            print(f"[ERROR] No objects detected in {img_path}")
    
    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print(" YOLOX SUMMARY")
        print(f"{'='*60}")
        print(f"🔢 Total objects detected: {total_objects}")
        print(f" Results saved to: outputs/yolox_*")
        print(f" Analysis complete!")
        
        # Compare if we have both images
        if len(all_results) == 2:
            imgs = list(all_results.keys())
            obj1 = set(det["object"] for det in all_results[imgs[0]]["detections"])
            obj2 = set(det["object"] for det in all_results[imgs[1]]["detections"])
            
            common = obj1 & obj2
            diff1 = obj1 - obj2
            diff2 = obj2 - obj1
            
            print(f"\n COMPARISON:")
            print(f"   Common objects: {sorted(common)}")
            print(f"   Only in {os.path.basename(imgs[0])}: {sorted(diff1)}")
            print(f"   Only in {os.path.basename(imgs[1])}: {sorted(diff2)}")
    
    return all_results


def test_confusion_matrix(image_path: str = "test_photos/images/bedroom1.jpeg"):
    """Easy test function for confusion matrix - just call this!"""
    print(f" TESTING YOLOX CONFUSION MATRIX")
    print(f"Image: {image_path}")

    try:
        detector = YOLOXDetector()
        detections = detector.detect_objects(image_path)

        if detections:
            print(f"[OK] SUCCESS: Detected {len(detections)} objects")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            vis_path = f"outputs/yolox_{base_name}_comparison.jpg"
            print(f"  Comparison visualization saved: {vis_path}")
            return True
        else:
            print("[ERROR] No objects detected")
            return False

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False


def test_all_photos(test_photos_dir: str = "test_photos", use_gpu=True):
    """
    Test YOLOX on all images in the test_photos directory
    
    Args:
        test_photos_dir: Directory containing images/ and labels/ subdirectories
        use_gpu: Whether to use GPU if available
    
    Returns:
        dict: Summary results for all tested images
    """
    import time
    start_time = time.time()
    
    print("="*60)
    print("TESTING YOLOX ON ALL TEST PHOTOS")
    print("="*60)
    
    images_dir = os.path.join(test_photos_dir, "images")
    labels_dir = os.path.join(test_photos_dir, "labels")
    
    if not os.path.exists(images_dir):
        print(f"[ERROR] Images directory not found: {images_dir}")
        return {}
    
    if not os.path.exists(labels_dir):
        print(f"[ERROR] Labels directory not found: {labels_dir}")
        return {}
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(image_extensions)]
    image_files.sort()
    
    if not image_files:
        print(f"[ERROR] No image files found in: {images_dir}")
        return {}
    
    print(f"\nFound {len(image_files)} test images")
    print("-"*60)
    
    # Process each image
    all_results = {}
    successful_tests = 0
    failed_tests = 0
    
    for i, image_file in enumerate(image_files, 1):
        # Construct paths
        image_path = os.path.join(images_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        label_file = f"{base_name}.json"
        label_path = os.path.join(labels_dir, label_file)
        
        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"\n[{i}/{len(image_files)}] [WARNING]  Skipping {image_file} - no matching label file")
            failed_tests += 1
            continue
        
        print(f"\n[{i}/{len(image_files)}]  Testing: {image_file}")
        print(f"              Label: {label_file}")
        
        try:
            # Run YOLOX with confusion matrix analysis
            results = yolox(image_path, label_path, use_gpu=use_gpu, create_overlay=True)
            
            # Store results
            all_results[image_file] = {
                'results': results,
                'image_path': image_path,
                'label_path': label_path,
                'status': 'success'
            }
            
            # Print summary for this image
            print(f"   [OK] mAP: {results['mean_average_precision']:.4f}, "
                  f"mF1: {results['mean_f1_score']:.4f}, "
                  f"Unmatched: {results['unmatched_objects']}, "
                  f"Missing: {results['missing_objects']}")
            
            successful_tests += 1
            
        except Exception as e:
            print(f"   [ERROR] Failed: {str(e)}")
            all_results[image_file] = {
                'error': str(e),
                'image_path': image_path,
                'label_path': label_path,
                'status': 'failed'
            }
            failed_tests += 1
    
    # Print overall summary
    print("\n" + "="*60)
    print(" OVERALL TEST SUMMARY")
    print("="*60)
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    
    # Calculate average metrics across successful tests
    if successful_tests > 0:
        avg_map = sum(r['results']['mean_average_precision'] 
                     for r in all_results.values() 
                     if r['status'] == 'success') / successful_tests
        avg_f1 = sum(r['results']['mean_f1_score'] 
                    for r in all_results.values() 
                    if r['status'] == 'success') / successful_tests
        
        print(f"\nAverage Metrics Across All Tests:")
        print(f"  Average mAP: {avg_map:.4f}")
        print(f"  Average mF1: {avg_f1:.4f}")
        
        # Show per-image results
        print(f"\nPer-Image Results:")
        print("-"*60)
        for image_file, result_data in all_results.items():
            if result_data['status'] == 'success':
                res = result_data['results']
                print(f"{image_file:20s} | mAP: {res['mean_average_precision']:.4f} | "
                      f"mF1: {res['mean_f1_score']:.4f} | "
                      f"Unmatched: {res['unmatched_objects']:2d} | "
                      f"Missing: {res['missing_objects']:2d}")
    
    print("="*60)
    
    # Display total execution time
    elapsed_time = time.time() - start_time
    print(f"\nTotal Execution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Total Execution Time per image: {elapsed_time/len(image_files):.2f} seconds")
    print("="*60)
    
    return all_results

if __name__ == "__main__":
    import sys
    
    # Check if user wants to test all photos
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Test all photos in test_photos directory
        test_all_photos(test_photos_dir="test_photos", use_gpu=True)
    else:
        # Test single image (default behavior matching rfdetr.py interface)
        print("="*60)
        print("Testing YOLOX with model_tests.py interface")
        print("="*60)
        print("Tip: Run with '--all' to test all images in test_photos/")
        print("="*60)
        
        # Example usage matching rfdetr.py
        image_path = "test_photos/images/bedroom1.jpeg"
        reference_json_path = "test_photos/labels/bedroom1.json"
        
        # Run YOLOX with confusion matrix analysis
        results = yolox(image_path, reference_json_path, use_gpu=True, create_overlay=True)
        
        print("\n" + "="*60)
        print("Test Complete!")
        print("="*60)
        print("\nTo test all images, run: python yolox.py --all")
        print("="*60)
