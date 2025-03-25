import numpy as np

def iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # Compute intersection
    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def weighted_average(bboxes, scores):
    """Compute a refined bounding box using a weighted average approach."""
    scores = np.array(scores)
    bboxes = np.array(bboxes)

    # Compute weighted sum
    x1 = np.sum(bboxes[:, 0] * scores) / np.sum(scores)
    y1 = np.sum(bboxes[:, 1] * scores) / np.sum(scores)
    x2 = np.sum(bboxes[:, 2] * scores) / np.sum(scores)
    y2 = np.sum(bboxes[:, 3] * scores) / np.sum(scores)

    return [x1, y1, x2, y2]

def nms_regression(bboxes, scores, iou_threshold=0.5):
    """
    Performs Non-Maximum Suppression (NMS) with regression-based refinement.

    Parameters:
        bboxes (list of lists): Bounding boxes [[x1, y1, x2, y2], ...]
        scores (list): Confidence scores for each bounding box.
        iou_threshold (float): IoU threshold for merging boxes.

    Returns:
        refined_bboxes (list): List of refined bounding boxes.
    """
    if len(bboxes) == 0:
        return []

    # Sort by confidence score (high to low)
    indices = np.argsort(scores)[::-1]
    bboxes = np.array(bboxes)[indices]
    scores = np.array(scores)[indices]

    refined_bboxes = []
    while len(bboxes) > 0:
        best_box = bboxes[0]
        best_score = scores[0]

        overlapping_boxes = [best_box]
        overlapping_scores = [best_score]

        # Compare IoU with remaining boxes
        remaining_boxes = []
        remaining_scores = []
        for i in range(1, len(bboxes)):
            if iou(best_box, bboxes[i]) > iou_threshold:
                overlapping_boxes.append(bboxes[i])
                overlapping_scores.append(scores[i])
            else:
                remaining_boxes.append(bboxes[i])
                remaining_scores.append(scores[i])

        # Merge overlapping boxes using weighted regression
        refined_box = weighted_average(overlapping_boxes, overlapping_scores)
        refined_bboxes.append(refined_box)

        # Continue with remaining boxes
        bboxes = np.array(remaining_boxes)
        scores = np.array(remaining_scores)

    return refined_bboxes

# Example usage:
bboxes = [[10, 20, 50, 80], [12, 22, 48, 78], [100, 120, 150, 170]]
scores = [0.9, 0.85, 0.95]

refined_boxes = nms_regression(bboxes, scores, iou_threshold=0.5)
print("Refined Bounding Boxes:", refined_boxes)