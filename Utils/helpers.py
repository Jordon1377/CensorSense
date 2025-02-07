from Utils.boundingbox_class import BoundingBox


def IoU(b1: BoundingBox, b2: BoundingBox) -> float:
    b1_x2, b1_y2 = b1.x1 + b1.w, b1.y1 + b1.h
    b2_x2, b2_y2 = b2.x1 + b2.w, b2.y1 + b2.h

    inter_x1 = max(b1.x1, b2.x1)
    inter_y1 = max(b1.y1, b2.y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = b1.w * b1.h
    b2_area = b2.w * b2.h
    union_area = b1_area + b2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0