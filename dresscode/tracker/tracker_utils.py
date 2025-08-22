import cv2

def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def bboxes_intersect(bbox1, bbox2):
    x1_a, y1_a, x2_a, y2_a = bbox1
    x1_b, y1_b, x2_b, y2_b = bbox2
    return (x1_a < x2_b and x2_a > x1_b and
            y1_a < y2_b and y2_a > y1_b)


def find_closest_hand(accessory_det, detections):
    acc_box = accessory_det['xyxy']
    for det in detections:
        if det['cls'] in ('hand', 'glove'):
            if bboxes_intersect(acc_box, det['xyxy']):
                return det['id']
    return None

def create_detections_dict(model, boxes, conf_thr):
    detections = []
    for box in boxes:
        conf = box.conf
        conf = float(conf.item()) if conf is not None else 0
        if conf < conf_thr:
            continue

        cls_idx = box.cls
        cls_idx = int(cls_idx.item())
        cls_name = model.names[cls_idx]

        xyxy = box.xyxy
        if xyxy is None:
            continue
        xy = xyxy[0].numpy().astype(int)
        x1, y1, x2, y2 = map(int, xy)

        tid = box.id
        tid = int(tid.item()) if tid is not None else None

        detections.append({
            'id': tid,
            'cls': cls_name,
            'conf': conf,
            'xyxy': (x1, y1, x2, y2),
        })
    return detections

def print_bbox(detections, frame):
    for d in detections:
        x1, y1, x2, y2 = d['xyxy']
        label = d['cls'] + f" ID:{d['id']}"
        if d['cls'] in ('watch', 'ring', 'bracelet'):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif d['cls'] == 'glove':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)