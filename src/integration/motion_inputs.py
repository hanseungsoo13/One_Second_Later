"""Prepare scene, crop, and pose inputs for PRW motion prediction."""
import numpy as np
from PIL import Image

from osl.pose import relative_pose


def _motion_sample(image, box, output_size, person_crop_cfg):
    width, height = image.size
    out_h, out_w = output_size
    scale = min(out_w / width, out_h / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (out_w - resized_w) // 2, (out_h - resized_h) // 2
    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (out_w, out_h))
    canvas.paste(resized, (pad_x, pad_y))
    pixels = np.asarray(canvas, dtype=np.float32) / 255
    pixels = (pixels - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)

    x1, y1, x2, y2 = map(float, box)
    crop_pixels = None
    if person_crop_cfg is not None:
        factor = person_crop_cfg['scale']
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        crop = image.crop((
            int(np.floor(center_x - (x2 - x1) * factor / 2)),
            int(np.floor(center_y - (y2 - y1) * factor / 2)),
            int(np.ceil(center_x + (x2 - x1) * factor / 2)),
            int(np.ceil(center_y + (y2 - y1) * factor / 2)),
        ))
        crop_h, crop_w = person_crop_cfg['size']
        crop_scale = min(crop_w / crop.width, crop_h / crop.height)
        crop_w_resized = max(1, round(crop.width * crop_scale))
        crop_h_resized = max(1, round(crop.height * crop_scale))
        crop = crop.resize((crop_w_resized, crop_h_resized), Image.Resampling.BILINEAR)
        crop_canvas = Image.new('RGB', (crop_w, crop_h))
        crop_canvas.paste(crop, ((crop_w - crop_w_resized) // 2, (crop_h - crop_h_resized) // 2))
        crop_pixels = np.asarray(crop_canvas, dtype=np.float32) / 255
        crop_pixels = (crop_pixels - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    transformed_box = np.array([
        x1 * resized_w / width + pad_x,
        y1 * resized_h / height + pad_y,
        x2 * resized_w / width + pad_x,
        y2 * resized_h / height + pad_y,
    ], dtype=np.float32)
    return (pixels.transpose(2, 0, 1), transformed_box,
            crop_pixels.transpose(2, 0, 1) if crop_pixels is not None else None,
            (resized_w / width, resized_h / height), (pad_x, pad_y))


def _predict_relative_poses(estimator, image, boxes):
    """Run frozen RTMPose on current-frame boxes and return bbox-relative [N,17,3]."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not len(boxes):
        return np.empty((0, 17, 3), dtype=np.float32)
    bgr = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
    xyxy = np.column_stack((boxes[:, :2], boxes[:, 2:]))
    points, scores = estimator(bgr, xyxy.tolist())
    poses = []
    for point, score, box in zip(points, scores, boxes):
        x1, y1, x2, y2 = box
        poses.append(relative_pose(point, score, [x1, y1, x2 - x1, y2 - y1]))
    if len(poses) != len(boxes):
        raise RuntimeError('RTMPose returned a different number of poses than input boxes')
    return np.stack(poses)
