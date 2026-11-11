def iou(box_a, box_b):
    """IoU of two boxes in [left, top, width, height] format. Returns a float in [0, 1]."""

    # 1. tlwh -> corners
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = box_a[0] + box_a[2], box_a[1] + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = box_b[0] + box_b[2], box_b[1] + box_b[3]

    # 2. overlap rectangle: later start, earlier end
    left = max(ax1, bx1)
    right = min(ax2, bx2)
    top = max(ay1, by1)
    bottom = min(ay2, by2)

    # 3. negative means "no overlap" -> use 0
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)

    # 4-5. intersection and union
    intersection = width * height
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - intersection

    # 6. guard against division by zero
    if union <= 0:
        return 0.0

    # 7. the answer
    return intersection / union