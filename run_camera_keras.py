import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import warnings
from ultralytics import YOLO

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
# You must change this if your downloaded model is named differently
MODEL_PATH = r"C:\Users\PAVILION 14\Downloads\trained_action_recognition_model.keras"

TARGET_CLASSES = [
    'waving_hand', 'using_mobile', 'talking', 'eating', 
    'drinking_water', 'clapping', 'barbell_biceps_curl', 'ball_throw'
]
OBJECT_CLASSES = [
    'none', 'cup', 'bottle', 'sports ball', 'cell phone', 
    'fork', 'spoon', 'knife', 'bowl', 'sandwich', 
    'remote', 'book', 'keyboard', 'laptop', 'scissors'
]

print("📦 Loading Neural Networks...")
# 1. Action Model
try:
    action_model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"❌ Failed to load action model at {MODEL_PATH}")
    print(e)
    exit()

# 2. YOLO Extractors
print("📦 Loading YOLO (this will download weights if not found locally)...")
model_pose = YOLO('yolov8n-pose.pt')
model_obj = YOLO('yolov8n.pt')
coco_names = model_obj.names
print("✅ Frameworks Ready!")

# -------------------------------------------------------------
# Real-Time Rolling Buffer logic
# -------------------------------------------------------------
pose_queue = deque(maxlen=64)      # Stores raw keypoints
active_obj = "none"                # Sliding active object
skip_frames = 0
FRAME_SKIP_RATE = 5                # Run heavy object detection every 5 frames

print("🚀 Starting Webcam! Press 'q' to quit.")
cap = cv2.VideoCapture(0)          # Switch to 1 if using an external webcam

# Increase resolution for better YOLO extraction if supported
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    orig_frame = frame.copy()
    
    # ---------------------------------------------------------
    # 1. Rapid Pose Extraction (Every Frame)
    # ---------------------------------------------------------
    results_pose = model_pose(frame, verbose=False)[0]
    
    if results_pose.keypoints is not None and len(results_pose.keypoints.data) > 0:
        kpts = results_pose.keypoints.data.cpu().numpy()[0][:, :2]
        
        # Display skeleton visually
        for k in kpts:
            if k[0] > 0 and k[1] > 0:
                cv2.circle(frame, (int(k[0]), int(k[1])), 5, (0, 255, 255), -1)
    else:
        kpts = np.zeros((17, 2))
        
    pose_queue.append(kpts)

    # ---------------------------------------------------------
    # 2. Interleaved Object Extraction
    # ---------------------------------------------------------
    skip_frames += 1
    if skip_frames >= FRAME_SKIP_RATE:
        skip_frames = 0
        results_obj = model_obj(frame, verbose=False)[0]
        detected = []
        if results_obj.boxes is not None:
            for cls_id in results_obj.boxes.cls.cpu().numpy():
                found_name = coco_names.get(int(cls_id), "unknown")
                if found_name in OBJECT_CLASSES:
                    detected.append(found_name)
                    
        # For real-time stability, just grab the most confident tracked object
        if len(detected) > 0:
            active_obj = detected[0]
        else:
            active_obj = "none"

    # ---------------------------------------------------------
    # 3. 89-Feature Mathematical Engine!
    # ---------------------------------------------------------
    action = "Waiting for frames..."
    confidence = 0.0

    if len(pose_queue) >= 5: # Need a few frames to run math
        
        # Build sequence-level Object Vector
        obj_vector = np.zeros(15)
        if active_obj in OBJECT_CLASSES:
            obj_vector[OBJECT_CLASSES.index(active_obj)] = 1.0
        else:
            obj_vector[0] = 1.0 # none
            
        feature_sequence = []
        prev_norm_kpts = None
        
        # Formulate features exactly as Colab does
        for i in range(len(pose_queue)):
            current_kpts = pose_queue[i]
            
            nose = current_kpts[0]
            l_wrist = current_kpts[9]
            r_wrist = current_kpts[10]
            l_shoulder = current_kpts[5]
            r_shoulder = current_kpts[6]
            
            # SCALE NORMALIZATION
            shoulder_dist = np.linalg.norm(l_shoulder - r_shoulder)
            scale = shoulder_dist if shoulder_dist > 0.01 else 1.0
            
            # NORMALIZED SKELETON
            norm_kpts = np.copy(current_kpts)
            if np.sum(current_kpts) > 0:
                norm_kpts = (norm_kpts - current_kpts[0]) / scale
            skeleton_flat = norm_kpts.flatten()
            
            # VELOCITY
            if prev_norm_kpts is None:
                velocity_flat = np.zeros(34)
            else:
                velocity_flat = (norm_kpts - prev_norm_kpts).flatten()
            prev_norm_kpts = norm_kpts
            
            # DISTANCE
            dist_l = np.linalg.norm(nose - l_wrist) / scale
            dist_r = np.linalg.norm(nose - r_wrist) / scale
            
            # DIRECTION
            dir_l = (nose - l_wrist) / scale
            dir_r = (nose - r_wrist) / scale
            
            # 89-DIMENSIONAL COMBINATION
            frame_matrix = np.concatenate([
                skeleton_flat, velocity_flat, obj_vector,
                [dist_l, dist_r], dir_l, dir_r
            ])
            feature_sequence.append(frame_matrix)
            
        # ---------------------------------------------------------
        # 4. Neural Network Prediction
        # ---------------------------------------------------------
        # Pad to exactly 64 length to meet GRU input requirements
        padded_features = np.zeros((1, 64, 89), dtype=np.float32)
        padded_features[0, :len(feature_sequence), :] = np.array(feature_sequence)
        
        probs = action_model.predict(padded_features, verbose=0)[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx] * 100
        action = TARGET_CLASSES[pred_idx]

    # ---------------------------------------------------------
    # 5. Heads-Up Display (HUD)
    # ---------------------------------------------------------
    # Draw Background Panel
    cv2.rectangle(frame, (10, 10), (450, 120), (0, 0, 0), -1)
    
    # Text
    cv2.putText(frame, f"Action: {action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"Conf: {confidence:.1f}%", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Target Obj: {active_obj}", (250, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
    cv2.putText(frame, f"Frames: {len(pose_queue)}/64", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.imshow("Activity Recognition (89-Feature GRU)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
