# Performance Audit

## Test Setup

| Parameter | Value |
|---|---|
| Machine | MacBook Pro (Apple Silicon) — x86 emulation via Rosetta |
| Python | 3.12.13 |
| CPU | (emulated) |
| RAM | (host dependent) |
| GPU | None (CPU only) |
| Test video | 1920×1080, 30 fps, 341 frames, 8.5 MB (shopping scene) |
| Output resolution | 1280×720 (limited by `MAX_WIDTH=1280`) |
| FRAME_SKIP | 2 (process every 3rd frame) |
| Models | YOLOv8n (detection), YOLOv8n-pose, Haar cascade (face) |
| Measurement | 341 frames, 114 tracked, benchmark via `time.perf_counter` + `tracemalloc` |

---

## Results

### Per-Component Timing (after optimizations)

| Component | Calls | Avg (ms) | Min (ms) | Max (ms) | Total (s) | % of pipeline |
|---|---|---|---|---|---|---|
| **Track Frame** (YOLOv8n + ByteTrack) | 341 | **188.5** | 162.7 | 236.6 | 64.3 | **60.0%** |
| **Face Detect** (Haar cascade @ 2x downscale) | 114 | **209.8** | 185.1 | 231.2 | 23.9 | **22.3%** |
| **Pose Detect** (YOLOv8n-pose) | 114 | **115.5** | 96.3 | 147.0 | 13.2 | **12.3%** |
| Read + Resize | 341 | 11.4 | 9.3 | 47.8 | 3.9 | 3.6% |
| Draw Boxes | 341 | 1.7 | 0.5 | 3.2 | 0.6 | 0.5% |
| Draw Skeleton | 341 | 1.9 | 0.01 | 9.1 | 0.7 | 0.6% |
| Draw Theft | 341 | 1.5 | 0.3 | 3.0 | 0.5 | 0.5% |
| Draw Faces | 341 | 0.03 | 0.01 | 0.2 | 0.01 | 0.01% |
| Video Write | 341 | 0.01 | 0.01 | 0.04 | 0.003 | 0.003% |
| Theft Update | 114 | 0.20 | 0.03 | 0.54 | 0.02 | 0.02% |

### Overall

| Metric | Value |
|---|---|
| Total pipeline time | **107.1 seconds** |
| Effective processing rate | **1.1 fps** |
| Real-time factor | **9.4×** (video is 3.8s, takes 107s to process) |
| Memory peak (tracemalloc) | **15.1 MB** |
| Memory steady-state | **6.1 MB** |
| Source length | 341 frames @ 30 fps = 11.4 seconds |
| Output length | Same (frame-accurate copy) |

---

## Bottleneck Breakdown

```
Track Frame  ████████████████████████████████████████████████████  60%
Face Detect  ████████████████████                                  22%
Pose Detect  ██████████                                            12%
Read+Resize  ███                                                    4%
Drawing      ██                                                     2%
Video Write  ▏                                                     <1%
```

### Tier 1 — Critical (>50ms per frame)
1. **Track Frame** (188ms) — YOLOv8n forward pass + ByteTrack association. This IS the pipeline; everything else is additive.
2. **Face Detect** (210ms) — OpenCV Haar cascade. Downsampled 2× already, down from 707ms.
3. **Pose Detect** (115ms) — YOLOv8n-pose forward pass. Separate model, separate inference.

### Tier 2 — Negligible (<5ms per frame)
- All drawing operations: ~5ms total
- Theft tracking logic: ~0.2ms
- Video encoding: ~0.01ms (writes are buffered)

---

## Optimizations Applied

### ✅ Face Detection 2× Downscale
- **Before**: 707ms — Haar cascade scanning full 1280×720
- **After**: 210ms — downsamples to max 640px wide before cascade
- **Speedup**: 3.4×
- **Tradeoff**: Minor accuracy loss on very small faces (<20px in downscaled frame). No impact on typical surveillance (faces are 80–200px).

### ✅ Removed Redundant YOLO Detection Call
- **Before**: `track_frame()` (376ms) + separate `detect_frame()` (191ms) = 567ms per tracked frame
- **After**: `track_frame()` already returns all 80 COCO classes; `detect_frame` call removed.
- **Speedup**: 191ms saved per tracked frame (no accuracy impact)
- **Note**: Tracked frames use `conf=0.3`, skip frames use `conf=0.5` (default). This slightly increases false positives on tracked frames but improves theft detection recall.

---

## Remaining Improvement Opportunities

| Optimization | Est. Speedup | Effort | Risk | Details |
|---|---|---|---|---|
| **ONNX export** | 2–3× on YOLO | Medium | Low | Convert `yolov8n.pt → yolov8n.onnx` (2–3× CPU inference speedup) |
| **Pose every 2nd tracked frame** | −8% total | Trivial | Low | Pose detection on every other tracked frame instead of every frame |
| **Face every 3rd tracked frame** | −7% total | Trivial | Low | Face detection on 1/3 of tracked frames (still get representative count) |
| **FRAME_SKIP=4** | −33% tracked frames | Trivial | Medium | Every 5th frame instead of every 3rd. Misses short interactions. |
| **MAX_WIDTH=640** | −25% on all CV ops | Trivial | Low | Smaller frames = faster resize, detect, draw. Lower quality output. |
| **Multi-processing** | N/A (single video) | High | High | Only helps with multiple concurrent cameras |
| **TensorRT** | 5–10× on GPU | High | High | Requires NVIDIA GPU, model conversion, deployment complexity |
| **Batch inference** | Minimal | High | High | Video is sequential; no benefit without multiple cameras |

### Estimated with All Low-Effort Optimizations

| Component | Current (ms) | Optimized (ms) | Notes |
|---|---|---|---|
| Track Frame | 188 | 63–94 | ONNX export (2–3×) |
| Face Detect | 210 | 70 | ONNX not helpful (Haar, not YOLO) |
| Pose Detect | 115 | 0–58 | Every 2nd frame; 58ms if ONNX |
| Drawing | 5 | 5 | Unchanged |
| **Tracked frame total** | **519ms (1.9 fps)** | **168ms (6.0 fps)** | **3.2× improvement** |
| **Skip frame total** | **188ms (5.3 fps)** | **63ms (15.9 fps)** | **3× improvement** |
| **Pipeline total (341 frames)** | **107s** | **~35s** | **3× faster** |

---

## Recommendations (Priority Order)

### 1. ONNX Export (High Impact, Low Effort)
```bash
pip install onnx onnxruntime
yolo export model=yolov8n.pt format=onnx
yolo export model=yolov8n-pose.pt format=onnx
```
Then modify `yolo_detector.py` to use the ONNX Runtime session:
```python
import onnxruntime as ort
sess = ort.InferenceSession("yolov8n.onnx")
# ... run inference, parse outputs
```
Expected speedup: **2–3× on all YOLO inference** (from 188ms → ~70ms per frame).

### 2. Reduce Pose Detection Frequency (Trivial Effort, Low Risk)
Run pose detection every 2nd tracked frame:
```python
if frame_idx % (settings.FRAME_SKIP + 1) == 0 and (frame_idx // 3) % 2 == 0:
```
Saves **~6s** (12% of current pipeline) with minimal behavioral analysis loss.

### 3. Reduce Face Detection Frequency (Trivial Effort, Low Risk)
Run face detection every 3rd tracked frame:
```python
if frame_idx % (settings.FRAME_SKIP + 1) == 0 and (frame_idx // 3) % 3 == 0:
```
Face count is already an aggregate; sampling is sufficient.

### 4. Tune FRAME_SKIP (Trivial Effort, Medium Risk)
- `FRAME_SKIP=4` → processes every 5th frame (40% fewer tracked frames, 40% faster)
- Risk: miss short theft interactions (<5 frames at 30fps = <0.17s)
- Best for: low-traffic surveillance where people move slowly

### 5. ONNX + TensorRT for Deployment (High Effort, High Impact)
Only if you deploy to a machine with an NVIDIA GPU:
```bash
yolo export model=yolov8n.pt format=engine device=0  # TensorRT
```
Expected speedup: **5–10× on GPU** (real-time 30fps processing).

---

## Memory Profile

| Resource | Usage |
|---|---|
| Python process RSS | ~120 MB (psutil) |
| tracemalloc peak | 15.1 MB (Python objects only) |
| YOLO model | ~12 MB (loaded in RAM) |
| YOLOv8-pose model | ~12 MB (loaded in RAM) |
| Haar cascade | ~1 MB |
| Video buffer | ~3.7 MB (1280×720×3) |
| Output video buffer | OS-managed (file write buffer) |
| SQLite DB | ~16 KB (negligible) |

**Total memory footprint: ~150 MB** (including Python interpreter, libraries, loaded models, frame buffer, and OS overhead).

---

## Benchmark Repeatability

To reproduce:
```bash
python tests/benchmark.py <video-path> [num-frames]
```

The benchmark script:
1. Warms up all models with one silent frame
2. Times each component with `time.perf_counter()` (ns precision)
3. Writes frames to `/dev/null` to isolate CPU from disk I/O
4. Uses `tracemalloc` for memory tracking
5. Reports per-component min/max/avg and pipeline summary

Run on the same video for comparable results. Variations of ±5% due to
CPU thermal throttling and system load are normal.

---

## History

| Date | Version | Pipeline Time | Effective fps | Real-time factor | Change |
|---|---|---|---|---|---|
| 2026-07-10 | Pre-opt | 183.6s | 0.6 | 16.15× | Baseline |
| 2026-07-10 | +Face 2× downscale | 127.5s | 0.9 | 11.21× | Face 3.4× faster |
| 2026-07-10 | +Remove redundant detect | 107.1s | 1.1 | 9.42× | −191ms/tracked frame |
| Next | +ONNX export | ~35s* | ~3.3 | ~3× | 3× YOLO speedup |
| Next | +Pose every 2nd frame | ~101s* | ~1.2 | ~8.9× | −12% pipeline time |
| Next | +FRAME_SKIP=4 | ~64s* | ~1.8 | ~5.6× | −40% tracked frames |

*Estimated — not yet implemented.
