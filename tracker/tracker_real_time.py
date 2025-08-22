import time
import cv2
import warnings
from collections import defaultdict
from ultralytics import YOLO
from tracker_utils import find_closest_hand, create_detections_dict, print_bbox

warnings.filterwarnings("ignore")

# ========== CONFIG ==========
MODEL_PATH = 'path/to/model'
TRACKER_YAML = 'bytetrack.yaml'
SOURCE = 0
DEVICE = 'cpu'
OUTPUT_VIDEO = None
CONF_THR = 0.35
FRAMES_REQUIRED = 5
VIOLATION_COOLDOWN_s = 60
CLASS_NAMES = ['hand', 'glove', 'watch', 'bracelet', 'ring']
# ============================

class HandState:
    def __init__(self):
        self.last_seen_t = 0.0
        self.last_violation_t = 0.0
        self.detected_history = {c: 0 for c in CLASS_NAMES}


class DresscodeMonitor:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.hand_state = dict()
        self.writer = None

    def dresscode_violations(self, hands_seen_by_id, acc_seen_by_id, v_time):
        frame_violations = []
        for id, st in self.hand_state.items():
            reasons = []

            if st.detected_history['hand'] >= FRAMES_REQUIRED:
                if (id in hands_seen_by_id) and ('glove' not in hands_seen_by_id.get(id, set())):
                    reasons.append("MANO SENZA GUANTO")

            present = []
            for c in ('watch', 'bracelet', 'ring'):
                if st.detected_history[c] >= FRAMES_REQUIRED and (c in acc_seen_by_id.get(id, set())):
                    present.append(c)
            if present:
                reasons.append("ACCESSORIO VIETATO: " + ", ".join(present))

            if reasons:
                if v_time - st.last_violation_t > VIOLATION_COOLDOWN_s:
                    st.last_violation_t = v_time
                frame_violations.append((id, reasons))
        return frame_violations

    def print_violations(self, frame, frame_violations, hands_seen_by_id, acc_seen_by_id):
        y_offset = 30
        for id, reasons in frame_violations:
            msg = f"VIOLAZIONE ID {id}: " + ", ".join(reasons)
            cv2.putText(frame, msg, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            y_offset += 28

        for h_id, state in self.hand_state.items():
            for hand_cls in ('hand', 'glove'):
                if hand_cls not in hands_seen_by_id.get(h_id, set()):
                    state.detected_history[hand_cls] = max(0, state.detected_history[hand_cls] - 1)

            for acc_cls in ('watch', 'bracelet', 'ring'):
                if acc_cls not in acc_seen_by_id.get(h_id, set()):
                    state.detected_history[acc_cls] = max(0, state.detected_history[acc_cls] - 1)

    def process(self):
        results = self.model.track(
            source=SOURCE,
            tracker=TRACKER_YAML,
            device=DEVICE,
            stream=True,
            verbose=False
        )

        for result in results:
            self._process_frame(result)

        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

    def _process_frame(self, result):

        frame = result.orig_img
        h, w = frame.shape[:2]
        
        if OUTPUT_VIDEO and self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 15, (w, h))

        boxes = result.boxes
        detections = create_detections_dict(self.model, boxes, CONF_THR)

        accessories_seen = []
        hands_seen_by_id = defaultdict(set)
        acc_seen_by_id = defaultdict(set)

        now = time.time()

        for det in detections:
            t_id = det['id']
            cls = det['cls']

            if t_id is not None:
                if cls in ('hand', 'glove'):
                    hands_seen_by_id[t_id].add(cls)
                    if t_id not in self.hand_state:
                        self.hand_state[t_id] = HandState()
                    self.hand_state[t_id].last_seen_t = now
                    self.hand_state[t_id].detected_history[cls] += 1

                elif cls in ('watch', 'ring', 'bracelet'):
                    closest_hand = find_closest_hand(det, detections)
                    if closest_hand is not None:
                        accessories_seen.append((closest_hand, cls))
                        acc_seen_by_id[closest_hand].add(cls)

        for id, state in self.hand_state.items():
            for hand, acc in accessories_seen:
                if id == hand:
                    state.detected_history[acc] += 1
        
        print_bbox(detections, frame)

        frame_violations = self.dresscode_violations(hands_seen_by_id, acc_seen_by_id, now)
        self.print_violations(frame, frame_violations, hands_seen_by_id, acc_seen_by_id)

        cv2.imshow("Dresscode Monitor", frame)
        if self.writer is not None:
            self.writer.write(frame)
        if cv2.waitKey(1) == ord('q'):
            exit()

        to_del = [id for id, s in self.hand_state.items() if now - s.last_seen_t > 30]
        for id in to_del:
            del self.hand_state[id]


if __name__ == '__main__':
    monitor = DresscodeMonitor()
    monitor.process()