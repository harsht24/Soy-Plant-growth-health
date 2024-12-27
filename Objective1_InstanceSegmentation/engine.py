import math
import sys
import torch
from utils import MetricLogger, SmoothedValue, reduce_dict
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) for two bounding boxes.
    box1, box2: [x_min, y_min, x_max, y_max]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def calculate_ap(ground_truths, predictions, iou_threshold):
    """
    Calculate Average Precision (AP) for a single image.
    Args:
        ground_truths: Array of ground truth boxes.
        predictions: Array of predicted boxes with confidence scores.
        iou_threshold: IoU threshold for true positive matching.

    Returns:
        AP (float): Average precision for this image.
    """
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)  # Sort by confidence score
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    matched_gt = set()

    for i, (pred_box, _) in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1
        for j, gt_box in enumerate(ground_truths):
            if j in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold:
            tp[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(float).eps)
    recall = tp_cumsum / len(ground_truths)

    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:]) if len(precision) > 1 else 0
    return ap

def evaluate(model, data_loader, device, iou_threshold=0.5, class_names=None):
    """
    Evaluate the model on a dataset.
    Args:
        model: The trained model (e.g., Mask R-CNN).
        data_loader: DataLoader for the dataset to evaluate.
        device: The device (e.g., 'cuda' or 'cpu').
        iou_threshold: IoU threshold for matching predictions to ground truth.
        class_names: Optional list of class names for detailed metrics.

    Returns:
        metrics: Dictionary containing evaluation metrics.
    """
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    all_ious = []
    all_ap = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)

        for target, output in zip(targets, outputs):
            true_boxes = target["boxes"].cpu().numpy()
            true_labels = target["labels"].cpu().numpy()

            pred_boxes = output["boxes"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            pred_scores = output["scores"].cpu().numpy()

            matched = set()
            predictions = [(pred_boxes[i], pred_scores[i]) for i in range(len(pred_boxes))]

            # Calculate AP for this image
            ap = calculate_ap(true_boxes, predictions, iou_threshold)
            all_ap.append(ap)


            for i, pred_box in enumerate(pred_boxes):
                if pred_scores[i] < 0.05:  # Skip low-confidence predictions
                    continue

                best_iou = 0
                best_idx = -1
                for j, true_box in enumerate(true_boxes):
                    if j in matched:
                        continue
                    iou = compute_iou(pred_box, true_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j

                if best_iou >= iou_threshold:
                    matched.add(best_idx)
                    all_true_labels.append(true_labels[best_idx])
                    all_pred_labels.append(pred_labels[i])
                    all_ious.append(best_iou)
                else:
                    all_pred_labels.append(pred_labels[i])
                    all_true_labels.append(-1)  # No match category

            # False negatives for unmatched ground truth boxes
            unmatched = set(range(len(true_boxes))) - matched
            for idx in unmatched:
                all_true_labels.append(true_labels[idx])
                all_pred_labels.append(-1)  # No detection category

    precision = precision_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)
    recall = recall_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)
    f1 = f1_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)
    mean_iou = np.mean(all_ious) if all_ious else 0
    map_score = np.mean(all_ap) if all_ap else 0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "mAP": map_score,
    }

    return metrics


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    num_batches = len(data_loader)

    batch_loss = 0
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Store loss for plotting
        batch_loss += loss_value
    
    final_loss = batch_loss / num_batches
    return final_loss