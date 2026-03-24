import cv2
import json
import os
import onnxruntime as ort
import numpy as np

def get_affine_transform(center, scale, rot, output_size, shift=(0., 0.), inv=False):
    """
    生成仿射变换矩阵 (保持宽高比的 Resize)
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = [0, src_w * -0.5]
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(dst, src)
    else:
        trans = cv2.getAffineTransform(src, dst)

    return trans

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class RacketPose:
    def __init__(self, model_path: str, input_size: tuple = (256,192),
                split_ratio: float = 2.0, padding: float = 1.25,
                providers: list[str] | None = None):
        if providers is None:
            providers = ['CPUExecutionProvider']
            try:
                import torch
                if torch.cuda.is_available():
                    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
                    if hasattr(os, "add_dll_directory") and os.path.isdir(torch_lib):
                        os.add_dll_directory(torch_lib)
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            except Exception:
                providers = ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception:
            if providers != ['CPUExecutionProvider']:
                self.session = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider'],
                )
            else:
                raise
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = input_size
        self.split_ratio = split_ratio
        self.padding = padding
        self.mean = (123.675, 116.28, 103.53)
        self.std = (58.395, 57.12, 57.375)

    def preprocess_bbox(self, img, bbox):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = max(0.0, min(x1, w - 1.0))
        y1 = max(0.0, min(y1, h - 1.0))
        x2 = max(0.0, min(x2, w - 1.0))
        y2 = max(0.0, min(y2, h - 1.0))

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        center = np.array([x1 + bw / 2.0, y1 + bh / 2.0], dtype=np.float32)

        aspect_ratio = self.input_size[0] / self.input_size[1]
        if bw > aspect_ratio * bh:
            scale_w = bw
            scale_h = bw / aspect_ratio
        else:
            scale_w = bh * aspect_ratio
            scale_h = bh
        scale = np.array([scale_w * self.padding, scale_h * self.padding], dtype=np.float32)

        trans = get_affine_transform(center, scale, 0, self.input_size)
        img_resized = cv2.warpAffine(img, trans, (self.input_size[0], self.input_size[1]))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32)
        img_tensor = img_tensor[None, ...]
        return img_tensor, center, scale

    def decode_simcc(self, simcc_x, simcc_y):
        """
        解码 SimCC Logits -> 坐标
        """
        # simcc_x: [B, K, W*ratio]
        # simcc_y: [B, K, H*ratio]
        
        # 1. Argmax
        x_locs = np.argmax(simcc_x, axis=2)
        y_locs = np.argmax(simcc_y, axis=2)
        
        # 2. 获取最大分数作为置信度
        max_val_x = np.max(simcc_x, axis=2)
        max_val_y = np.max(simcc_y, axis=2)
        scores = np.minimum(max_val_x, max_val_y)
        
        # 3. 还原到 input_size 尺度
        locs = np.stack([x_locs, y_locs], axis=-1).astype(np.float32)
        locs /= self.split_ratio
        
        return locs[0], scores[0] # 返回第一个样本 [K, 2], [K]


    def __call__(self, img, bbox):
        input_tensor, center, scale = self.preprocess_bbox(img, bbox)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        simcc_x, simcc_y = outputs
        kpts, scores = self.decode_simcc(simcc_x, simcc_y)
        trans_inv = get_affine_transform(center, scale, 0, self.input_size, inv=True)
        kpts_orig = np.zeros_like(kpts)
        for i in range(len(kpts)):
            kpts_orig[i] = affine_transform(kpts[i], trans_inv)
        return kpts_orig, scores
