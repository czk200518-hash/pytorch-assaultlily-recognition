import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ANIME_CASCADE_PATH = str(Path(__file__).parent / 'models' / 'lbpcascade_animeface.xml')

_CASCADE_TMP_PATH = os.path.join(
    tempfile.gettempdir(), 'anime_face_detector_cascade.xml'
)


def _nms_merge(faces, iou_threshold=0.3):
    """NMS 合并重叠的人脸检测框"""
    if len(faces) == 0:
        return np.zeros((0, 4), dtype=np.int32)

    boxes = np.array(faces, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = np.arange(len(boxes))
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

        idx = np.where(iou <= iou_threshold)[0]
        order = order[idx + 1]

    return boxes[keep].astype(np.int32)


def _imread(filepath):
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法读取图片文件: {filepath}')
    return image


def _imwrite(filepath, image):
    filepath = os.path.normpath(filepath)
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'):
        ext = '.png'
    success, data = cv2.imencode(ext, image)
    if not success:
        raise RuntimeError(f'图片编码失败: {filepath}')

    data = data.tobytes()
    with open(filepath, 'wb') as f:
        f.write(data)


def _load_anime_cascade():
    if not os.path.exists(ANIME_CASCADE_PATH):
        raise FileNotFoundError(
            f'动漫人脸级联模型文件未找到: {ANIME_CASCADE_PATH}\n'
            '请确保 models/lbpcascade_animeface.xml 存在'
        )

    if not os.path.exists(_CASCADE_TMP_PATH) or \
            os.path.getsize(_CASCADE_TMP_PATH) != os.path.getsize(ANIME_CASCADE_PATH):
        shutil.copy2(ANIME_CASCADE_PATH, _CASCADE_TMP_PATH)

    cascade = cv2.CascadeClassifier(_CASCADE_TMP_PATH)
    if cascade.empty():
        raise RuntimeError('无法加载动漫人脸级联模型文件')
    return cascade


def _preprocess_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    results = {}
    results['histeq'] = cv2.equalizeHist(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    results['clahe'] = clahe.apply(gray)

    clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    results['clahe_strong'] = clahe_strong.apply(gray)

    return results


def _detect_single_pass(cascade, gray, scale_factor, min_neighbors, min_size):
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces.tolist() if len(faces) > 0 else []


def _filter_faces(faces, image_h, image_w, edge_mask=None, strict=True):
    """智能过滤误检框

    过滤规则:
    1. 宽高比: 人脸接近正方形，0.55~1.35
    2. 位置: 对于全身立绘(高度>宽度×1.5)，人脸应在图片上半部分
    3. 大小: 人脸面积应在图像的 0.3% 到 35% 之间
    4. 边缘密度: 人脸区域比衣服/腿部纹理更丰富(严格模式)
    """
    if len(faces) == 0:
        return []

    img_area = image_h * image_w
    is_fullbody = image_h > image_w * 1.5

    filtered = []
    reasons = []

    for x, y, fw, fh in faces:
        aspect = fw / max(fh, 1)

        if aspect < 0.55 or aspect > 1.35:
            reasons.append(f'  排除: ({x},{y},{fw},{fh}) 宽高比={aspect:.2f} (合理范围 0.55~1.35)')
            continue

        face_area_ratio = (fw * fh) / img_area
        if face_area_ratio < 0.003 or face_area_ratio > 0.80:
            reasons.append(f'  排除: ({x},{y},{fw},{fh}) 面积比={face_area_ratio:.3f} (合理范围 0.003~0.80)')
            continue

        if is_fullbody:
            face_center_y = y + fh / 2
            if face_center_y > image_h * 0.72:
                reasons.append(f'  排除: ({x},{y},{fw},{fh}) 位置过低 (y重心={face_center_y:.0f}/{image_h})')
                continue

        if strict and edge_mask is not None and fw > 20 and fh > 20:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(image_w, x + fw)
            y2 = min(image_h, y + fh)
            roi = edge_mask[y1:y2, x1:x2]
            if roi.size > 0:
                edge_density = np.count_nonzero(roi) / roi.size
                if edge_density < 0.03:
                    reasons.append(f'  排除: ({x},{y},{fw},{fh}) 边缘密度过低={edge_density:.3f}')
                    continue

        filtered.append([x, y, fw, fh])

    return filtered, reasons


def _build_edge_mask(gray):
    """构建边缘密度掩码"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def detect_faces_advanced(image_path, min_size=(32, 32),
                          min_neighbors_low=3, min_neighbors_high=7,
                          iou_threshold=0.25, intensity='standard',
                          strict_filter=False):
    """
    增强版人脸检测：多轮检测 + NMS合并 + 智能误检过滤

    参数:
        intensity: 'fast' | 'standard' | 'thorough'
        strict_filter: 启用严格过滤（排除衣服/腿部等误检）
    """
    image = _imread(image_path)
    h, w = image.shape[:2]
    cascade = _load_anime_cascade()
    gray_dict = _preprocess_gray(image)

    all_faces = []

    if intensity == 'fast':
        scale_factors = [1.1]
        preprocessing_keys = ['clahe']
        mn_list = [min_neighbors_low]
    elif intensity == 'thorough':
        scale_factors = [1.01, 1.05, 1.1]
        preprocessing_keys = ['clahe', 'clahe_strong', 'histeq']
        mn_list = [min_neighbors_low, min_neighbors_low + 1, min_neighbors_high]
    else:
        scale_factors = [1.05, 1.1]
        preprocessing_keys = ['clahe', 'clahe_strong']
        mn_list = [min_neighbors_low, min_neighbors_low + 1]

    for pp_key in preprocessing_keys:
        gray = gray_dict[pp_key]
        for sf in scale_factors:
            for mn in mn_list:
                faces = _detect_single_pass(cascade, gray, sf, mn, min_size)
                all_faces.extend(faces)

    merged = _nms_merge(all_faces, iou_threshold=iou_threshold)

    if strict_filter and len(merged) > 0:
        gray_for_edges = gray_dict['clahe']
        edge_mask = _build_edge_mask(gray_for_edges)
        filtered, reasons = _filter_faces(merged, h, w, edge_mask, strict=True)
        if reasons:
            print('[严格过滤] 排除了以下误检:')
            for r in reasons:
                print(r)
        merged = np.array(filtered) if filtered else np.zeros((0, 4), dtype=np.int32)

    return merged


def detect_faces(image_path, min_size=(32, 32),
                 min_neighbors_low=3, min_neighbors_high=7,
                 intensity='standard', iou_threshold=0.25,
                 strict_filter=False):
    faces = detect_faces_advanced(
        image_path, min_size=min_size,
        min_neighbors_low=min_neighbors_low,
        min_neighbors_high=min_neighbors_high,
        iou_threshold=iou_threshold,
        intensity=intensity,
        strict_filter=strict_filter,
    )
    return faces.tolist() if len(faces) > 0 else []


def crop_faces(image_path, output_dir,
               min_size=(32, 32),
               min_neighbors_low=3, min_neighbors_high=7,
               padding=0, intensity='standard',
               export_mode='separate',
               strict_filter=False):
    """检测并裁剪人脸

    export_mode:
        'separate' — 以原图名称创建子文件夹，内含原图副本 + 截取人脸
        'merged'   — 所有截取人脸直接保存在 output_dir 下，不创建子文件夹
    """

    image = _imread(image_path)
    h, w = image.shape[:2]

    faces = detect_faces_advanced(
        image_path, min_size=min_size,
        min_neighbors_low=min_neighbors_low,
        min_neighbors_high=min_neighbors_high,
        intensity=intensity,
        strict_filter=strict_filter,
    )

    image_path_obj = Path(image_path)
    base_name = image_path_obj.stem
    src_ext = image_path_obj.suffix

    if export_mode == 'merged':
        save_dir = output_dir
    else:
        save_dir = os.path.join(output_dir, base_name)

    os.makedirs(save_dir, exist_ok=True)

    if export_mode == 'separate':
        original_save_path = os.path.join(save_dir, f'原图_{base_name}{src_ext}')
        shutil.copy2(image_path, original_save_path)

    results = []

    for i, (x, y, fw, fh) in enumerate(faces):
        x = int(x)
        y = int(y)
        fw = int(fw)
        fh = int(fh)

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + fw + padding)
        y2 = min(h, y + fh + padding)

        face_img = image[y1:y2, x1:x2]

        if face_img.size == 0:
            continue

        face_filename = f'截取人脸_{i+1}_{base_name}{src_ext}'
        save_path = os.path.join(save_dir, face_filename)
        _imwrite(save_path, face_img)

        results.append({
            'index': i + 1,
            'x': x,
            'y': y,
            'width': fw,
            'height': fh,
            'cropped_x': x1,
            'cropped_y': y1,
            'cropped_width': x2 - x1,
            'cropped_height': y2 - y1,
            'save_path': save_path,
        })

    return results


def draw_boxes(image_path, output_path=None, min_size=(32, 32),
               min_neighbors_low=3, min_neighbors_high=7,
               intensity='standard', color=(0, 255, 255),
               thickness=2, strict_filter=False):
    image = _imread(image_path)

    faces = detect_faces_advanced(
        image_path, min_size=min_size,
        min_neighbors_low=min_neighbors_low,
        min_neighbors_high=min_neighbors_high,
        intensity=intensity,
        strict_filter=strict_filter,
    )

    for i, (x, y, fw, fh) in enumerate(faces):
        x, y, fw, fh = int(x), int(y), int(fw), int(fh)
        cv2.rectangle(image, (x, y), (x + fw, y + fh), color, thickness)
        label = f'Face #{i+1}'
        cv2.putText(image, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        _imwrite(output_path, image)

    return image, len(faces)


def batch_detect_folder(folder_path, output_dir=None,
                        min_size=(32, 32),
                        min_neighbors_low=3, min_neighbors_high=7,
                        padding=0, intensity='standard',
                        progress_callback=None,
                        export_mode='separate',
                        strict_filter=False):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(folder_path), 'detected_faces')

    folder = Path(folder_path)
    results = {}
    total_faces = 0

    all_images = [
        p for p in sorted(folder.rglob('*'))
        if p.suffix.lower() in image_extensions
    ]
    total_images = len(all_images)

    for idx, img_path in enumerate(all_images):
        try:
            if progress_callback:
                progress_callback(idx + 1, total_images)

            faces = crop_faces(
                str(img_path), output_dir,
                min_size=min_size,
                min_neighbors_low=min_neighbors_low,
                min_neighbors_high=min_neighbors_high,
                padding=padding,
                intensity=intensity,
                export_mode=export_mode,
                strict_filter=strict_filter,
            )

            rel_path = str(img_path.relative_to(folder))
            results[rel_path] = faces
            total_faces += len(faces)

            if faces:
                print(f'  [{rel_path}] 检测到 {len(faces)} 个人脸')
            else:
                print(f'  [{rel_path}] 未检测到人脸')

        except Exception as e:
            print(f'  [{img_path.name}] 处理失败: {e}')

    print(f'\n共处理 {len(results)} 张图片，检测到 {total_faces} 个人脸')
    print(f'输出目录: {output_dir}')
    return results, total_faces
