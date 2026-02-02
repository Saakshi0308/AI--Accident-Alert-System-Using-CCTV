import os, time, datetime
import cv2
import numpy as np
from ultralytics import YOLO
import requests
import warnings
from roboflow import Roboflow

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
VIDEO_PATH = 1  # Can be file path or webcam index (0, 1, etc.)
OUTPUT_VIDEO = r"D:\python codes\accident/telegram_accident_output.mp4"
SAVE_DIR = 'accident_frames'
DISPLAY_VIDEO = True
SAVE_OUTPUT_VIDEO = True

# ---------------- TELEGRAM CONFIGURATION ----------------
# 🔒 Replace with your actual Telegram Bot Token and Chat ID
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"  # e.g., "12758735890:ABCDEF..."
CHAT_ID = "YOUR_TELEGRAM_CHAT_ID_HERE"      # e.g., "35450445525"
TELEGRAM_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

# ---------------- ROBOFLOW CONFIGURATION ----------------
# 🔒 Replace with your own Roboflow credentials and project details
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY_HERE"
ROBOFLOW_WORKSPACE = "YOUR_WORKSPACE_NAME_HERE"
ROBOFLOW_PROJECT = "YOUR_PROJECT_NAME_HERE"
ROBOFLOW_VERSION = 1  # Update if needed

# ---------------- DETECTION PARAMETERS ----------------
MODEL_PATH = 'yolov8n.pt'  # You can use a custom YOLO model if needed
CONFIDENCE = 0.5
NOTIFICATION_COOLDOWN = 60
PROCESS_EVERY_N_FRAMES = 2

os.makedirs(SAVE_DIR, exist_ok=True)
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)
model.overrides['verbose'] = False
vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}

print("[INFO] Initializing Roboflow...")
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    crash_model = project.version(ROBOFLOW_VERSION).model
    print("[INFO] Roboflow crash detection model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load Roboflow model: {e}")
    crash_model = None


# ---------------- CLASSES ----------------
class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.chat_id = chat_id
        self.last_notification_time = 0

    def send_emergency_alert(self, accidents):
        now = time.time()
        if now - self.last_notification_time < NOTIFICATION_COOLDOWN:
            return
        self.last_notification_time = now

        message = f"🚨 ACCIDENT ALERT 🚨\nTotal Vehicles Involved: {len(accidents)}"
        for acc in accidents:
            message += f"\n- Type: {acc.get('type', 'vehicle')} | Confidence: {acc.get('confidence', 0):.2f}"

        try:
            requests.post(self.url, data={"chat_id": self.chat_id, "text": message})
            print("[INFO] Telegram alert sent")
        except Exception as e:
            print(f"[ERROR] Telegram alert failed: {e}")


class OptimizedVehicleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, detections):
        updated_tracks = {}
        for det in detections:  # det = (x1, y1, x2, y2, class_name)
            self.tracks[self.next_id] = {
                "bbox": det[:4],
                "class": det[4],
                "is_moving": True,
                "lost": 0
            }
            updated_tracks[self.next_id] = self.tracks[self.next_id]
            self.next_id += 1
        return updated_tracks


class OptimizedAccidentAnalyzer:
    def __init__(self, crash_model=None):
        self.crash_model = crash_model

    def analyze_accidents(self, frame, tracks, frame_number):
        accidents = []
        # Simplified logic: mark every 80th frame as accident for testing
        if frame_number % 80 == 0 and len(tracks) > 0:
            for tid, track in tracks.items():
                accidents.append({
                    "bbox": track["bbox"],
                    "confidence": 0.9,
                    "type": track["class"]
                })
        return accidents


# ---------------- VISUALIZATION ----------------
def draw_optimized_visualization(frame, tracks, accidents, frame_number):
    for track_id, track in tracks.items():
        x1, y1, x2, y2 = map(int, track['bbox'])
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track_id}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    for accident in accidents:
        x1, y1, x2, y2 = map(int, accident['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        label = f"ACCIDENT! ({accident['confidence']:.1f})"
        cv2.putText(frame, label, (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    info_text = f"Frame: {frame_number} | Tracks: {len(tracks)}"
    if accidents:
        info_text += f" | ACCIDENTS: {len(accidents)}"
    cv2.putText(frame, info_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


# ---------------- MAIN ----------------
def main():
    if not os.path.exists(str(VIDEO_PATH)) and not isinstance(VIDEO_PATH, int):
        print(f"[ERROR] Video file not found: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    notifier = TelegramNotifier(BOT_TOKEN, CHAT_ID)
    tracker = OptimizedVehicleTracker()
    analyzer = OptimizedAccidentAnalyzer(crash_model)

    video_writer = None
    if SAVE_OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc,
                                       fps // PROCESS_EVERY_N_FRAMES,
                                       (width, height))

    frame_number, processed_frames, total_accidents = 0, 0, 0
    processing_times = []

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        if frame_number % PROCESS_EVERY_N_FRAMES != 0:
            continue
        processed_frames += 1

        detections = []
        try:
            results = model(frame, conf=CONFIDENCE, imgsz=416, verbose=False)[0]
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                for box, cls in zip(boxes, classes):
                    class_name = model.names[int(cls)]
                    if class_name in vehicle_classes:
                        x1, y1, x2, y2 = map(int, box)
                        if (x2 - x1) * (y2 - y1) > 1000:
                            detections.append((x1, y1, x2, y2, class_name))
        except Exception as e:
            print(f"[WARNING] Detection failed on frame {frame_number}: {e}")

        tracks = tracker.update(detections)
        accidents = analyzer.analyze_accidents(frame, tracks, frame_number)

        if accidents:
            total_accidents += len(accidents)
            notifier.send_emergency_alert(accidents)
            print(f"[ACCIDENT] Frame {frame_number}: {len(accidents)} accident(s)")

        frame = draw_optimized_visualization(frame, tracks, accidents, frame_number)

        if video_writer:
            video_writer.write(frame)
        if DISPLAY_VIDEO:
            display_frame = frame
            if frame.shape[1] > 1280:
                scale = 1280 / frame.shape[1]
                display_frame = cv2.resize(frame, (int(frame.shape[1] * scale),
                                                   int(frame.shape[0] * scale)))
            cv2.imshow("Optimized Accident Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        elapsed = time.time() - start_time
        processing_times.append(elapsed)

    avg_time = np.mean(processing_times) if processing_times else 0
    avg_fps = 1 / avg_time if avg_time > 0 else 0
    print("[INFO] Processing complete.")
    print(f"[STATS] Total frames processed: {processed_frames}")
    print(f"[STATS] Total accidents detected: {total_accidents}")
    print(f"[STATS] Average FPS: {avg_fps:.2f}")

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting...")
        cv2.destroyAllWindows()
