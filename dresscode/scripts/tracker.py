"""
Dresscode real-time tracker + verifier (YOLOv8 + ByteTrack)
- supports: classes = ['mano','guanto','orologio','bracciale','anello']
- If model includes 'person', we map hands -> person using bbox containment.
- Logs violations to CSV and optionally writes annotated video.

Usage:
    python dresscode_realtime_tracker.py
"""
import time
import csv
import math
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# ========== CONFIG ==========
MODEL_PATH = r'ultralytics/training_output/training_03/weights/best.pt'  # tuo modello
TRACKER_YAML = 'bytetrack.yaml'   # tracker config file (must exist)
SOURCE = 0                        # 0=webcam, or 'video.mp4' o RTSP url
DEVICE = 'cpu'                    # 'cpu' or '0' for GPU
OUTPUT_VIDEO = 'dresscode_out.mp4'  # None per non salvare
LOG_CSV = 'violazioni.csv'

CONF_THR = 0.35                   # soglia confidence per considerare detection
FRAMES_REQUIRED = 6               # occorrenze consecutive per confermare violazione
VIOLATION_COOLDOWN_s = 10         # cooldown in secondi per non ripetere log stesso ID
MAX_ASSIGN_DIST = 120             # px distanza massima per associare mano->persona
# Assumi che model.names contenga i nomi delle classi; altrimenti modifica qui:
# Esempio di nomi nel tuo training (modificali se diversi)
CLASS_NAMES = ['mano', 'guanto', 'orologio', 'bracciale', 'anello']  # personalizza

# quale label indica la persona (se presente nel tuo modello), altrimenti None
PERSON_CLASS_NAME = 'person'  # metti None se non presente nel tuo set
# ============================


def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def iou(boxA, boxB):
    # boxes: (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom > 0 else 0.0


# State per persona (o persona-proxy)
class PersonState:
    def __init__(self):
        self.frame_seen = 0
        self.last_seen_t = 0.0
        self.detected_history = defaultdict(int)  # class_name -> consecutive frames seen
        self.last_violation_t = 0.0


def main():
    # carica modello
    model = YOLO(MODEL_PATH)

    # apri CSV log
    csv_file = open(LOG_CSV, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'tracked_id', 'violation', 'details'])

    # video writer (opzionale)
    writer = None
    if OUTPUT_VIDEO:
        # inizializziamo writer dopo primo frame (per dimensioni)
        writer = None

    # tracking in stream -> generator of Results
    results = model.track(source=SOURCE, tracker=TRACKER_YAML, device=DEVICE, stream=True, verbose=False)

    person_states = dict()   # id -> PersonState
    frame_idx = 0
    t0 = time.time()

    for r in results:
        frame_idx += 1
        # r è un oggetto Results (uno per frame)
        # otteniamo immagine annotata o originale:
        try:
            frame = r.orig_img  # numpy array BGR (preferibile)
        except Exception:
            # fallback
            frame = r.plot()  # immagine con annotazioni

        h, w = frame.shape[:2]

        # inizializza writer quando possibile
        if OUTPUT_VIDEO and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20, (w, h))

        # Parse detections
        # r.boxes: Boxes object; ogni box ha .xyxy, .cls, .conf, .id (se tracking)
        detections = []
        for box in r.boxes:
            conf = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else 1.0
            if conf < CONF_THR:
                continue
            cls_idx = int(box.cls.cpu().numpy())
            cls_name = None
            # map class idx to name: prefer model.names if available
            if hasattr(model, 'names') and cls_idx in model.names:
                cls_name = model.names[cls_idx]
            else:
                # fallback to our CLASS_NAMES if lengths match
                cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)

            xy = box.xyxy[0].cpu().numpy().astype(int)  # array([x1,y1,x2,y2])
            track_id = None
            try:
                track_id = int(box.id.cpu().numpy())
            except Exception:
                # box.id potrebbe non esistere
                track_id = None

            detections.append({
                'id': track_id,
                'cls': cls_name,
                'conf': conf,
                'xyxy': tuple(int(x) for x in xy)
            })

        # Build indexes: persons, hands, accessories
        persons = []
        hands = []
        accessories = []
        for d in detections:
            if PERSON_CLASS_NAME is not None and d['cls'] == PERSON_CLASS_NAME:
                persons.append(d)
            elif d['cls'] in ('mano', 'hand', 'mano_real'):   # try common hand names
                hands.append(d)
            elif d['cls'] in ('guanto', 'glove'):
                hands.append(d)   # glove considered with hand class for mapping (we will detect presence)
            elif d['cls'] in ('orologio','watch'):
                accessories.append(d)
            elif d['cls'] in ('bracciale','bracelet'):
                accessories.append(d)
            elif d['cls'] in ('anello','ring'):
                accessories.append(d)
            else:
                # if unknown class name but present in our CLASS_NAMES mapping
                if d['cls'] in CLASS_NAMES:
                    if d['cls'] == 'mano':
                        hands.append(d)
                    elif d['cls'] == 'guanto':
                        hands.append(d)
                    else:
                        accessories.append(d)

        # If persons exist, assign hands/accessories to persons using containment/closest center
        assignments = defaultdict(list)  # person_id -> list of detection dicts
        if persons:
            # compute person bbox and id (use track id if available otherwise assign index)
            for p in persons:
                pid = p['id'] if p['id'] is not None else f"person_idx_{persons.index(p)}"
                assignments[pid] = []
            # helper to find containing person
            for det in hands + accessories:
                cx, cy = bbox_center(det['xyxy'])
                best_pid = None
                best_dist = float('inf')
                for p in persons:
                    px1, py1, px2, py2 = p['xyxy']
                    if px1 <= cx <= px2 and py1 <= cy <= py2:
                        # contained -> prefer immediately
                        best_pid = p['id'] if p['id'] is not None else f"person_idx_{persons.index(p)}"
                        break
                    else:
                        # distance to person center
                        pcx, pcy = bbox_center(p['xyxy'])
                        dist = math.hypot(cx - pcx, cy - pcy)
                        if dist < best_dist:
                            best_dist = dist
                            best_pid = p['id'] if p['id'] is not None else f"person_idx_{persons.index(p)}"
                # only assign if reasonably close
                if best_dist < MAX_ASSIGN_DIST or (px1 <= cx <= px2 and py1 <= cy <= py2):
                    assignments[best_pid].append(det)
        else:
            # No persons: create person-proxies grouping hands by proximity (simple clustering)
            # We'll assign groups by nearest neighbor: each tracked ID becomes a proxy if available
            proxy_map = {}
            for det in hands + accessories:
                # if detection already has a track id, use it as proxy id
                if det['id'] is not None:
                    pid = det['id']
                    assignments[pid].append(det)
                else:
                    # form new proxy by center quantization
                    cx, cy = bbox_center(det['xyxy'])
                    found = False
                    for pid, members in assignments.items():
                        # compute dist to first member center
                        m = members[0]
                        mcx, mcy = bbox_center(m['xyxy'])
                        if math.hypot(cx - mcx, cy - mcy) < MAX_ASSIGN_DIST:
                            assignments[pid].append(det)
                            found = True
                            break
                    if not found:
                        # new proxy id
                        proxy_id = f"proxy_{len(assignments)}"
                        assignments[proxy_id].append(det)

        # Update person states and check violations
        now = time.time()
        frame_violations = []  # collect to draw/log
        for pid, dets in assignments.items():
            # ensure state exists
            if pid not in person_states:
                person_states[pid] = PersonState()
            state = person_states[pid]
            state.frame_seen = frame_idx
            state.last_seen_t = now

            # Reset temporary marker for this frame
            seen = {cls: 0 for cls in CLASS_NAMES + [PERSON_CLASS_NAME] if cls}

            # mark seen classes in this frame for this pid
            for d in dets:
                cls = d['cls']
                # map synonyms
                if cls in ('glove',):
                    cls = 'guanto'
                if cls in ('watch',):
                    cls = 'orologio'
                if cls in ('bracelet',):
                    cls = 'bracciale'
                if cls in ('ring',):
                    cls = 'anello'
                if cls in ('hand',):
                    cls = 'mano'

                if cls in seen:
                    seen[cls] = seen.get(cls, 0) + 1

            # update consecutive frame counters
            for cls_name in seen:
                if seen[cls_name] > 0:
                    state.detected_history[cls_name] += 1
                else:
                    state.detected_history[cls_name] = 0

            # Dresscode logic:
            # Condition 1: if mano detected and guanto not detected for FRAMES_REQUIRED consecutive frames -> violation
            vio_reasons = []
            if state.detected_history.get('mano', 0) >= FRAMES_REQUIRED:
                if state.detected_history.get('guanto', 0) < FRAMES_REQUIRED:
                    vio_reasons.append('mano_senza_guanto')

            # Condition 2: if accessory detected (orologio/bracciale/anello) for FRAMES_REQUIRED frames -> violation
            if any(state.detected_history.get(c, 0) >= FRAMES_REQUIRED for c in ('orologio', 'bracciale', 'anello')):
                present = [c for c in ('orologio', 'bracciale', 'anello') if state.detected_history.get(c,0) >= FRAMES_REQUIRED]
                vio_reasons.append('accessorio_vietato:' + ",".join(present))

            # register violation if any and not in cooldown
            if vio_reasons:
                if now - state.last_violation_t > VIOLATION_COOLDOWN_s:
                    state.last_violation_t = now
                    # log
                    ts = time.time() - t0
                    csv_writer.writerow([f"{ts:.2f}", pid, ";".join(vio_reasons), str(dets)])
                    csv_file.flush()
                # prepare to draw on frame
                frame_violations.append((pid, vio_reasons))

        # Draw detections and violations on frame
        # draw all detected boxes
        for d in detections:
            x1,y1,x2,y2 = d['xyxy']
            cls = d['cls']
            tid = d['id']
            color = (0,255,0)
            label = f"{cls}"
            if tid is not None:
                label += f" ID:{tid}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # draw violations prominently
        y_offset = 30
        for pid, reasons in frame_violations:
            text = f"VIOLAZIONE {pid}: " + ", ".join(reasons)
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            y_offset += 28

        # show + save
        cv2.imshow("Dresscode Monitor", frame)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # cleanup old person states (not seen for a long time)
        to_del = [pid for pid, s in person_states.items() if now - s.last_seen_t > 60]
        for pid in to_del:
            del person_states[pid]

    # fine stream
    if writer:
        writer.release()
    csv_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
