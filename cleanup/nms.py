import torch
import torchvision

def xywh2xyxy(x):
    """
    Convert bounding box format from (x, y, w, h) to (x1, y1, x2, y2).
    """
    assert x.shape[-1] == 4, f"Expected last dim=4, got {x.shape}"
    y = x.new_empty(x.shape)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=300,
    pre_nms_topk=1000
):
    """
    Optimized NMS for pose estimation.
    Args:
        prediction (torch.Tensor): Tensor with shape (batch, num_detections, detection_size)
                                   where detection_size >= 56 (first 4: x,y,w,h, 5th: conf, rest: keypoints).
        conf_thres (float): Confidence threshold.
        iou_thres (float): IoU threshold for NMS.
        max_det (int): Maximum detections per image.
        pre_nms_topk (int): Limit candidates to the top-K detections.
    Returns:
        List[torch.Tensor]: List of tensors (one per batch image) after NMS.
    """

    # Convert bounding boxes from (x, y, w, h) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = []
    for pred in prediction:
        # Filter out detections below confidence threshold
        pred = pred[pred[:, 4] > conf_thres]
        if pred.shape[0] == 0:
            output.append(torch.empty((0, 56), device=pred.device))
            continue

        # Limit to top candidates to reduce NMS computation
        if pred.shape[0] > pre_nms_topk:
            scores = pred[:, 4]
            _, idx = scores.topk(pre_nms_topk)
            pred = pred[idx]
        boxes = pred[:, :4]
        scores = pred[:, 4]
        keep = torchvision.ops.nms(boxes, scores, iou_thres)
        keep = keep[:max_det]  # Cap detections per image
        output.append(pred[keep])
    return output
