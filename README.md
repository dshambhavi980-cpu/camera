# Real-Time Activity Recognition (89-Feature GRU)

Live webcam activity recognition that combines **YOLOv8 pose estimation**, **YOLOv8 object
detection**, and a **trained GRU model** to classify what a person is doing, frame by frame.

Instead of feeding raw pixels to the network, each frame is reduced to an 89-dimensional
feature vector built from the skeleton, its motion, and the object the person is interacting
with. A rolling 64-frame buffer of those vectors is passed to the GRU for prediction.

## Recognized actions

`waving_hand` · `using_mobile` · `talking` · `eating` · `drinking_water` · `clapping` ·
`barbell_biceps_curl` · `ball_throw`

## How it works

1. **Pose extraction (every frame)** — `yolov8n-pose` returns 17 COCO keypoints, drawn on the
   frame as a live skeleton overlay.
2. **Object extraction (every 5th frame)** — `yolov8n` detects nearby objects and keeps the
   most relevant one from a 15-class whitelist (cup, bottle, cell phone, sports ball, …).
   Interleaving keeps the heavier detector off the per-frame hot path.
3. **Feature engineering (89 dims per frame)**
   - `34` normalized skeleton coordinates — recentered on the nose, scaled by shoulder distance
     so the model is invariant to distance from the camera
   - `34` velocity values — frame-to-frame delta of the normalized skeleton
   - `15` one-hot active-object vector
   - `2` nose→wrist distances (left, right)
   - `4` nose→wrist direction vectors (left, right)
4. **Prediction** — the buffer is zero-padded to `(1, 64, 89)` and passed to the Keras GRU,
   which outputs a class and confidence.
5. **HUD** — action, confidence, active object, and buffer fill are drawn over the video feed.

## Files

| File | Description |
| --- | --- |
| `run_camera_keras.py` | Full real-time inference pipeline (pose → objects → features → GRU → HUD) |
| `active_objects.pkl` | Cached per-sequence active-object labels used during dataset preparation |

## Requirements

```bash
pip install opencv-python numpy tensorflow ultralytics
```

A trained model file (`trained_action_recognition_model.keras`) is required and is **not**
included in this repository. YOLO weights (`yolov8n-pose.pt`, `yolov8n.pt`) download
automatically on first run.

## Usage

Set `MODEL_PATH` near the top of `run_camera_keras.py` to point at your trained model:

```python
MODEL_PATH = r"path/to/trained_action_recognition_model.keras"
```

Then run:

```bash
python run_camera_keras.py
```

Press **`q`** to quit.

## Notes

- Uses camera index `0` by default. For an external webcam, change `cv2.VideoCapture(0)` to `1`.
- Capture resolution is requested at 1280×720; the camera falls back to its nearest supported mode.
- Predictions begin once at least 5 frames are buffered, and stabilize as the buffer approaches 64.
- Feature construction must match the training pipeline exactly — changing the order or scaling
  of any block will silently degrade accuracy.
