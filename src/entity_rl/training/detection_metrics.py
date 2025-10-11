"""
Detection accuracy metrics for MOT-based ENROS training.

This module provides detection evaluation metrics including IoU, precision, recall,
and mean Average Precision (mAP) calculations.
"""

from typing import List, Tuple, Dict, Union
import numpy as np
import torch
from torch import Tensor


def calculate_iou(boxes1: Union[Tensor, np.ndarray], boxes2: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """
    Calculate Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1: First set of bounding boxes in format [x1, y1, x2, y2] or [N, 4]
        boxes2: Second set of bounding boxes in format [x1, y1, x2, y2] or [M, 4]

    Returns:
        IoU matrix of shape [N, M] where iou[i,j] is IoU between boxes1[i] and boxes2[j]
    """
    if isinstance(boxes1, np.ndarray):
        boxes1 = torch.from_numpy(boxes1).float()
    if isinstance(boxes2, np.ndarray):
        boxes2 = torch.from_numpy(boxes2).float()

    # Ensure boxes are 2D
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)

    N = boxes1.shape[0]
    M = boxes2.shape[0]

    # Calculate intersection
    # boxes1: [N, 4] -> [N, 1, 4] -> [N, M, 4]
    # boxes2: [M, 4] -> [1, M, 4] -> [N, M, 4]
    boxes1_expanded = boxes1.unsqueeze(1).expand(N, M, 4)
    boxes2_expanded = boxes2.unsqueeze(0).expand(N, M, 4)

    # Calculate intersection coordinates
    intersection_x1 = torch.max(boxes1_expanded[:, :, 0], boxes2_expanded[:, :, 0])
    intersection_y1 = torch.max(boxes1_expanded[:, :, 1], boxes2_expanded[:, :, 1])
    intersection_x2 = torch.min(boxes1_expanded[:, :, 2], boxes2_expanded[:, :, 2])
    intersection_y2 = torch.min(boxes1_expanded[:, :, 3], boxes2_expanded[:, :, 3])

    # Calculate intersection area
    intersection_width = torch.clamp(intersection_x2 - intersection_x1, min=0)
    intersection_height = torch.clamp(intersection_y2 - intersection_y1, min=0)
    intersection_area = intersection_width * intersection_height

    # Calculate areas of both boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    # Expand to [N, M]
    area1_expanded = area1.unsqueeze(1).expand(N, M)
    area2_expanded = area2.unsqueeze(0).expand(N, M)

    # Calculate union area
    union_area = area1_expanded + area2_expanded - intersection_area

    # Calculate IoU, avoiding division by zero
    iou = intersection_area / torch.clamp(union_area, min=1e-6)

    return iou


def calculate_precision_recall_f1(
    pred_boxes: Union[Tensor, np.ndarray],
    gt_boxes: Union[Tensor, np.ndarray],
    iou_threshold: float = 0.5,
    pred_scores: Union[Tensor, np.ndarray] = None,
    score_threshold: float = 0.0
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1-score for object detection.

    Args:
        pred_boxes: Predicted bounding boxes [N, 4] in format [x1, y1, x2, y2]
        gt_boxes: Ground truth bounding boxes [M, 4] in format [x1, y1, x2, y2]
        iou_threshold: IoU threshold for considering a detection as correct
        pred_scores: Optional confidence scores for predictions [N]
        score_threshold: Minimum score threshold for considering predictions

    Returns:
        Dictionary containing precision, recall, and F1-score
    """
    if isinstance(pred_boxes, np.ndarray):
        pred_boxes = torch.from_numpy(pred_boxes).float()
    if isinstance(gt_boxes, np.ndarray):
        gt_boxes = torch.from_numpy(gt_boxes).float()

    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'true_positives': 0,
            'false_positives': int(pred_boxes.shape[0]) if pred_boxes.numel() > 0 else 0,
            'false_negatives': int(gt_boxes.shape[0]) if gt_boxes.numel() > 0 else 0
        }

    # Filter predictions by score threshold if provided
    if pred_scores is not None:
        if isinstance(pred_scores, np.ndarray):
            pred_scores = torch.from_numpy(pred_scores).float()
        valid_idx = pred_scores >= score_threshold
        pred_boxes = pred_boxes[valid_idx]
        pred_scores = pred_scores[valid_idx]

    if pred_boxes.numel() == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': int(gt_boxes.shape[0])
        }

    # Calculate IoU matrix
    iou_matrix = calculate_iou(pred_boxes, gt_boxes)  # [N_pred, N_gt]

    # For each prediction, find the best matching ground truth
    best_iou_per_pred, best_gt_idx = torch.max(iou_matrix, dim=1)  # [N_pred]

    # Count true positives (predictions with IoU >= threshold)
    true_positive_mask = best_iou_per_pred >= iou_threshold
    true_positives = torch.sum(true_positive_mask).item()
    false_positives = torch.sum(~true_positive_mask).item()

    # Count ground truths that were matched (to calculate false negatives)
    matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    for i, is_tp in enumerate(true_positive_mask):
        if is_tp:
            matched_gt[best_gt_idx[i]] = True

    false_negatives = torch.sum(~matched_gt).item()

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def calculate_ap(
    pred_boxes: Union[Tensor, np.ndarray],
    pred_scores: Union[Tensor, np.ndarray],
    gt_boxes: Union[Tensor, np.ndarray],
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate Average Precision (AP) for a single IoU threshold.

    Args:
        pred_boxes: Predicted bounding boxes [N, 4]
        pred_scores: Confidence scores for predictions [N]
        gt_boxes: Ground truth bounding boxes [M, 4]
        iou_threshold: IoU threshold for positive detections

    Returns:
        Average Precision value
    """
    if isinstance(pred_boxes, np.ndarray):
        pred_boxes = torch.from_numpy(pred_boxes).float()
    if isinstance(pred_scores, np.ndarray):
        pred_scores = torch.from_numpy(pred_scores).float()
    if isinstance(gt_boxes, np.ndarray):
        gt_boxes = torch.from_numpy(gt_boxes).float()

    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return 0.0

    # Sort predictions by confidence score (descending)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes_sorted = pred_boxes[sorted_indices]
    pred_scores_sorted = pred_scores[sorted_indices]

    # Calculate IoU matrix
    iou_matrix = calculate_iou(pred_boxes_sorted, gt_boxes)

    # Track which ground truths have been matched
    gt_matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)

    # Lists to store precision and recall at each threshold
    precisions = []
    recalls = []

    true_positives = 0
    false_positives = 0

    for i in range(pred_boxes_sorted.shape[0]):
        # Find best matching ground truth for this prediction
        best_iou, best_gt_idx = torch.max(iou_matrix[i], dim=0)

        if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
            # True positive
            true_positives += 1
            gt_matched[best_gt_idx] = True
        else:
            # False positive
            false_positives += 1

        # Calculate precision and recall at this point
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / gt_boxes.shape[0]

        precisions.append(precision)
        recalls.append(recall)

    # Convert to tensors
    precisions = torch.tensor(precisions)
    recalls = torch.tensor(recalls)

    # Calculate AP using the 11-point interpolation method
    ap = 0.0
    for t in torch.arange(0.0, 1.1, 0.1):
        # Find precisions for recalls >= t
        valid_recalls = recalls >= t
        if torch.any(valid_recalls):
            ap += torch.max(precisions[valid_recalls]).item()

    return ap / 11.0


def calculate_map(
    pred_boxes: Union[Tensor, np.ndarray],
    pred_scores: Union[Tensor, np.ndarray],
    gt_boxes: Union[Tensor, np.ndarray],
    iou_thresholds: List[float] = None
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) across multiple IoU thresholds.

    Args:
        pred_boxes: Predicted bounding boxes [N, 4]
        pred_scores: Confidence scores for predictions [N]
        gt_boxes: Ground truth bounding boxes [M, 4]
        iou_thresholds: List of IoU thresholds to evaluate at

    Returns:
        Dictionary containing mAP and AP values for each threshold
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    ap_values = []
    results = {}

    for threshold in iou_thresholds:
        ap = calculate_ap(pred_boxes, pred_scores, gt_boxes, threshold)
        ap_values.append(ap)
        results[f'AP@{threshold:.2f}'] = ap

    # Calculate mean AP
    results['mAP'] = np.mean(ap_values)
    results['AP@0.5'] = ap_values[0] if len(ap_values) > 0 else 0.0  # Standard COCO AP

    return results


def evaluate_detection_batch(
    pred_boxes_batch: List[Union[Tensor, np.ndarray]],
    pred_scores_batch: List[Union[Tensor, np.ndarray]],
    gt_boxes_batch: List[Union[Tensor, np.ndarray]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate detection performance on a batch of samples.

    Args:
        pred_boxes_batch: List of predicted bounding boxes for each sample
        pred_scores_batch: List of prediction scores for each sample
        gt_boxes_batch: List of ground truth bounding boxes for each sample
        iou_threshold: IoU threshold for evaluation

    Returns:
        Dictionary containing aggregated metrics across the batch
    """
    batch_metrics = {
        'precision': [],
        'recall': [],
        'f1_score': [],
        'ap': []
    }

    for pred_boxes, pred_scores, gt_boxes in zip(pred_boxes_batch, pred_scores_batch, gt_boxes_batch):
        # Calculate precision/recall for this sample
        pr_metrics = calculate_precision_recall_f1(pred_boxes, gt_boxes, iou_threshold, pred_scores)
        batch_metrics['precision'].append(pr_metrics['precision'])
        batch_metrics['recall'].append(pr_metrics['recall'])
        batch_metrics['f1_score'].append(pr_metrics['f1_score'])

        # Calculate AP for this sample
        ap = calculate_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold)
        batch_metrics['ap'].append(ap)

    # Calculate mean metrics across batch
    return {
        'mean_precision': np.mean(batch_metrics['precision']),
        'mean_recall': np.mean(batch_metrics['recall']),
        'mean_f1_score': np.mean(batch_metrics['f1_score']),
        'mean_ap': np.mean(batch_metrics['ap']),
        'std_precision': np.std(batch_metrics['precision']),
        'std_recall': np.std(batch_metrics['recall']),
        'std_f1_score': np.std(batch_metrics['f1_score']),
        'std_ap': np.std(batch_metrics['ap'])
    }


def convert_mot_format_to_xyxy(mot_boxes: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """
    Convert MOT format bounding boxes [x, y, w, h] to [x1, y1, x2, y2] format.

    Args:
        mot_boxes: Bounding boxes in MOT format [N, 4] where each box is [x, y, w, h]

    Returns:
        Bounding boxes in [x1, y1, x2, y2] format
    """
    if isinstance(mot_boxes, np.ndarray):
        is_numpy = True
        mot_boxes = torch.from_numpy(mot_boxes).float()
    else:
        is_numpy = False

    if mot_boxes.numel() == 0:
        return mot_boxes

    xyxy_boxes = torch.zeros_like(mot_boxes)
    xyxy_boxes[:, 0] = mot_boxes[:, 0]  # x1 = x
    xyxy_boxes[:, 1] = mot_boxes[:, 1]  # y1 = y
    xyxy_boxes[:, 2] = mot_boxes[:, 0] + mot_boxes[:, 2]  # x2 = x + w
    xyxy_boxes[:, 3] = mot_boxes[:, 1] + mot_boxes[:, 3]  # y2 = y + h

    return xyxy_boxes.numpy() if is_numpy else xyxy_boxes


def extract_detection_data_from_mot_sample(
    graph_data,
    reward: int,
    agent_pos: Tuple[float, float],
    image_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract bounding box and confidence data from MOT graph sample.

    Args:
        graph_data: Graph data containing node features
        reward: Reward value (not used for detection extraction)
        agent_pos: Agent position
        image_size: Image dimensions (width, height)

    Returns:
        Tuple of (boxes, scores) where boxes are in [x1, y1, x2, y2] format
    """
    if hasattr(graph_data, 'x') and graph_data.x is not None:
        node_features = graph_data.x.numpy() if isinstance(graph_data.x, torch.Tensor) else graph_data.x

        # Extract bounding box coordinates from node features
        # Assuming node features contain [x, y, w, h, ...] in first 4 dimensions
        if node_features.shape[1] >= 4:
            boxes_xywh = node_features[:, :4]  # [x, y, w, h]

            # Convert from normalized coordinates to pixel coordinates
            boxes_xywh[:, [0, 2]] *= image_size[0]  # x, w
            boxes_xywh[:, [1, 3]] *= image_size[1]  # y, h

            # Convert to [x1, y1, x2, y2] format
            boxes_xyxy = convert_mot_format_to_xyxy(boxes_xywh)

            # Extract confidence scores if available (assuming they're in the 5th column)
            if node_features.shape[1] >= 5:
                scores = node_features[:, 4]
            else:
                # If no confidence scores, use uniform confidence
                scores = np.ones(boxes_xyxy.shape[0])

            return boxes_xyxy, scores

    # Return empty arrays if no valid data found
    return np.array([]).reshape(0, 4), np.array([])