# AI Video Intelligence Platform — Complete Product Roadmap

## Vision
A sellable AI-powered surveillance & analytics platform for retail stores,
supermarkets, restaurants, and cafés. Real-time detection of theft,
suspicious behavior, occupancy, queues, and customer flow.

---

## Phase 0 — Current Foundation (Done)
- [x] Video upload (file + URL)
- [x] YOLOv8 object detection (person + 80 COCO classes)
- [x] ByteTrack person tracking (unique IDs across frames)
- [x] Face detection (Haar cascade, cyan boxes)
- [x] Theft detection (unattended objects + person-object interaction)
- [x] Item concealment (person overlaps item → item disappears → THEFT alert)
- [x] Annotated video output (H.264)
- [x] Web UI (FastAPI + Jinja2)
- [x] `.gitignore`

**Gap vs TheftGuard AI (reference repo):**
They have pose estimation, ROI drawing, face rec with blacklist,
Next.js dashboard, audio alerts, multi-threaded cameras.
We have item tracking, concealment logic, video upload analysis.

---

## Phase 1 — Sellable MVP (8-10 weeks)

### 1.1 Pose Estimation Engine (1 week)
YOLOv8-pose to detect human keypoints (shoulders, elbows, wrists, hips, knees).

**Implementation:**
```python
model = YOLO("yolov8n-pose.pt")
results = model(frame)
keypoints = results[0].keypoints  # 17 keypoints per person
```

**Detectable behaviors:**

| Behavior | Keypoint logic | Value |
|---|---|---|
| **Hand-to-pocket** | wrist y → hip y (hand moving toward pocket) | Potential concealment |
| **Hand-to-bag** | wrist x,y → bag bbox overlap | Putting item in bag |
| **Bending down** | shoulder y → hip y distance shrinks | Hiding items in low shelves |
| **Reaching shelf** | wrist x,y → shelf zone overlap | Item selection tracking |
| **Wrist in restricted** | wrist x,y inside cash-register zone | Theft at counter |
| **Fighting** | two people close, arms flailing | Violence alert |

**Test:**
- Upload video of person putting phone in pocket → flag "hand-to-pocket"
- Upload video of person bending to tie shoe → flag "bending"
- Verify keypoints overlay on annotated video (skeleton lines)

### 1.2 Modern React/Next.js Dashboard (2-3 weeks)
Replace Jinja2 templates with a proper frontend:

**Tech stack:** Next.js 14, TypeScript, Tailwind CSS, Recharts, WebSocket

**Pages:**
```
/                          → Login
/dashboard                 → Live camera grid + KPIs
/dashboard/camera/:id      → Single camera live view + zone editor
/dashboard/history         → Alert log + CSV export + filtering
/dashboard/settings        → Alert config, camera management, face blacklist
/dashboard/analytics       → Heatmaps, trends, hourly traffic
```

**Key UI components from TheftGuard:**
- Glassmorphic dark theme
- CPU/RAM telemetry bars
- Weekly trend charts (Recharts)
- Alert history with search/filter by camera, type, date
- CSV export of alerts
- Audio siren via Web Audio API on theft alert

**Migration plan:**
- Keep FastAPI as backend (all AI processing stays)
- FastAPI serves JSON API + WebSocket for live data
- Next.js frontend is a separate `dashboard/` directory
- Docker Compose runs both

### 1.3 Interactive ROI Zone Drawer (1 week)
HTML5 Canvas tool to draw security zones on the live video feed.

**Features:**
- Draw polygons directly on video frame
- Scale coordinates from canvas to 1280x720 frame resolution
- Store zones per camera in `cameras.json`
- Zone types:
  - `restricted` — wrist intrusion triggers alarm
  - `entry_exit` — count people entering/leaving
  - `shelf` — track item selection, detect empty periods
  - `checkout` — queue length + wait time
  - `seating` — table occupancy (café/restaurant)

**Test:**
- Draw a restricted zone around a cash register
- Wave hand over it → trigger zone intrusion alert
- Draw entry zone → 3 people walk through → count = 3

### 1.4 Facial Recognition with Blacklist/Whitelist (1 week)

**Implementation:**
- Use `face_recognition` library (dlib CNN) or lightweight `insightface`
- Enroll faces from uploaded photos → store encodings in SQLite
- In video: detect face, compute encoding, match against database
- Categories: `blacklist` (alarm), `whitelist` (green badge), `unknown`

**Face Management UI:**
- Upload portrait photo
- Assign name + category (blacklist/whitelist/VIP)
- View enrolled faces grid
- One-click delete

**Privacy:**
- Toggle to blur all non-whitelist faces
- GDPR compliant: auto-delete encodings after X days
- Option to run fully offline (no cloud)

### 1.5 Multi-Threaded Camera Pipeline (1 week)

**Architecture:**
```text
Camera thread 1 → FrameQueue → Processing Pool → Results → WebSocket → Dashboard
Camera thread 2 → FrameQueue →                        ↑
Camera thread N → FrameQueue →─────────────────────────┘
```

- Each camera gets its own thread for frame capture (RTSP/webcam)
- Frames pushed to a shared processing queue
- Worker pool consumes queue, runs YOLO, sends results
- Camera isolation: track IDs prefixed with `{camera_id}_{track_id}`
- Config: FPS limit per camera, resolution scaling

**Camera sources:**
- USB webcam (index 0, 1, 2...)
- RTSP/ONVIF (`rtsp://user:pass@ip:554/stream1`)
- Local file (loop playback for demo)
- AI webcam (ESP32-CAM HTTP stream)

### 1.6 Alert System with Multiple Channels (1 week)

| Alert Type | Trigger | Channels |
|---|---|---|
| **Concealment** | Hand-to-pocket detected | Dashboard, Audio siren, Telegram |
| **Zone intrusion** | Wrist in restricted zone | Dashboard, Audio siren |
| **Theft** | Item concealed (overlap + disappear) | Telegram with screenshot, Email |
| **Loitering** | Person in zone > threshold | Dashboard, Email digest |
| **Queue long** | Queue > threshold for >30s | Dashboard banner, Telegram |
| **Max occupancy** | People count > limit | Dashboard, Email |
| **Unattended object** | Object alone > 30s | Dashboard, Audio |
| **Fighting** | Aggressive pose detected | Urgent: Telegram + Email + Siren |
| **Blacklisted face** | Known thief detected | Urgent alert with screenshot |

**Implementation:**
- Telegram bot (python-telegram-bot)
- SMTP email with screenshot attachment
- Web Audio API siren (client-side, no audio file needed)
- WebSocket push to dashboard
- Alert cooldown (3s between same-type alerts to prevent spam)

### 1.7 Activity Heatmaps (3-4 days)
- Accumulate person centroid positions per camera
- Normalize to frame size
- Overlay color gradient (blue=cold, red=hot) on video
- Export as PNG overlay

**Test:**
- Let 50 people walk through frame → heatmap shows walking paths in red
- Identify bottleneck zones where heatmap is brightest

---

## Phase 2 — Testing & Demo Kit

### 2.1 Test Fixture Videos
Record these scenarios for demo and automated testing:

| Scenario | Content | Used for |
|---|---|---|
| `theft_hand_to_pocket.mp4` | Person picks up phone, puts in pocket | Pose concealment test |
| `theft_bag_conceal.mp4` | Person puts item into backpack | Theft detection test |
| `theft_bending.mp4` | Person bends down, hides item | Bending detection test |
| `entry_exit_3.mp4` | 3 people walk through door | Zone counting test |
| `queue_5_people.mp4` | 5 people lined up at counter | Queue detection test |
| `fighting_2.mp4` | 2 people shoving each other | Fight detection test |
| `loitering_2min.mp4` | 1 person standing in aisle 2min | Loitering test |
| `faces_5_known.mp4` | 5 known people walk past camera | Face recognition test |
| `blacklisted_face.mp4` | Blacklisted person enters store | Blacklist alert test |

Create by staging with friends/family, or download from Pixabay/Pexels
and composite with FFmpeg.

### 2.2 Automated Test Suite

```bash
pytest tests/ -v
```

| Test file | Coverage |
|---|---|
| `test_pose.py` | Keypoint detection, hand-to-pocket, bending |
| `test_theft.py` | Concealment triggers, unattended objects |
| `test_zones.py` | Point-in-polygon, entry/exit counts |
| `test_faces.py` | Face detection, encoding, blacklist match |
| `test_alerts.py` | Alert rules fire at correct thresholds |
| `test_camera.py` | RTSP/webcam frame capture, FPS monitoring |
| `test_dashboard_api.py` | All REST endpoints return correct JSON |

### 2.3 Demo Mode
- Built-in demo that loops a pre-recorded MP4 with staged theft events
- Auto-generates alerts every 30s for dashboard walkthrough
- One command: `docker compose up` → fully functional demo
- Sample customer data seeded in SQLite

---

## Phase 3 — Deployment & Packaging

### 3.1 Docker Compose
```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes: ["./data:/data"]
    devices: ["/dev/video0:/dev/video0"]  # for webcam passthrough
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]  # NVIDIA GPU passthrough

  dashboard:
    build: ./dashboard
    ports: ["3000:3000"]
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000

  postgres:
    image: postgres:16
    volumes: ["./pgdata:/var/lib/postgresql/data"]
```

### 3.2 Edge Device Images
Pre-built SD card images for:
- Raspberry Pi 5 (2-3 FPS on CPU)
- Orange Pi 5 (5-8 FPS with NPU)
- Intel N100 Mini PC (10-15 FPS)
- Jetson Orin Nano (25-30 FPS with TensorRT)

### 3.3 ONNX + TensorRT Optimization
```bash
yolo export model=yolov8n.pt format=onnx  # 2-3x CPU speedup
yolo export model=yolov8n.pt format=engine device=0  # TensorRT for Jetson
```

---

## Phase 4 — Pricing & Go-to-Market

### Pricing Tiers

| Tier | Price | Cameras | Features |
|---|---|---|---|
| **Starter** | $29/mo | 1-2 | Person counting, face detection, basic alerts |
| **Pro** | $79/mo | 4-6 | + Pose detection, zone editor, theft alerts, Telegram |
| **Enterprise** | $199/mo | Unlimited | + Face blacklist, heatmaps, multi-camera, API, on-prem |

**One-time hardware bundle:**
- Raspberry Pi 5 Kit + Camera + SD card with pre-installed software: **$299**
- Intel N100 Mini PC + 4 USB cameras: **$899**

### Target Customers
| Segment | Pain point | Our solution |
|---|---|---|
| **Convenience stores** | Shoplifting losses | Concealment detection, blacklist face alerts |
| **Supermarkets** | Queue management, shelf monitoring | Queue KPIs, shelf emptiness alerts |
| **Restaurants/Cafés** | Table turnover, wait times | Table occupancy, queue wait time |
| **Retail chains** | Multi-store visibility | Multi-camera dashboard, heatmaps |
| **Warehouses** | Loitering, restricted areas | Zone intrusion, loitering alerts |

### Sales Demo Checklist
- [ ] Live webcam showing person tracking in real-time
- [ ] Theft demo: put phone in pocket → alert fires on dashboard
- [ ] Zone demo: draw restricted area → wave hand → alarm sounds
- [ ] Face demo: enroll face as blacklist → show alert when detected
- [ ] Queue demo: 3 people line up → dashboard shows queue length
- [ ] Heatmap demo: walk around → see trail on heatmap
- [ ] Dashboard: show CPU/RAM, weekly trends, CSV export

---

## Phase 5 — Advanced Features

| Feature | Description | Value |
|---|---|---|
| **Re-identification** | Track people across multiple cameras via appearance embedding | Full customer journey |
| **POS integration** | Link queue wait time to register transaction speed | Staff scheduling |
| **SKU-level tracking** | Detect specific products by fine-tuning YOLO on store inventory | Exact item theft alerts |
| **Cloud dashboard** | Multi-store remote viewing | Chain operations |
| **Mobile app** | Push alerts, live view on phone | Manager on-the-go |
| **Analytics API** | REST API for third-party BI tools (Tableau, PowerBI) | Enterprise integration |
| **Body camera** | Wearable AI camera for security guards | Mobile patrol |
| **LPR** | License plate recognition for parking lots | Vehicle theft |

---

## Week-by-Week Execution Plan

| Week | Work | Deliverable |
|---|---|---|
| **1** | Pose estimation (YOLOv8-pose + keypoint logic) | Hand-to-pocket, bending, wrist detection |
| **2-3** | Next.js dashboard scaffold + WebSocket | Live feed page with camera grid |
| **4** | ROI zone drawer (Canvas tool) | Draw zones, store to JSON, zone events |
| **5** | Face recognition + blacklist/whitelist | Enroll faces, auto-alert on blacklist match |
| **6** | Multi-threaded camera pipeline | 4 concurrent RTSP/webcam streams |
| **7** | Alert system (Telegram, Email, Audio) | Alerts fire + notification delivery |
| **8** | Heatmaps + activity analytics | Per-camera heat overlay |
| **9** | Test fixtures + automated tests | CI pipeline passes |
| **10** | Docker packaging + demo mode | One-command demo deployment |

## Immediate Next Step
Implement **Pose Estimation** (YOLOv8-pose). It unlocks the most
sellable features: hand-to-pocket concealment detection, bending analysis,
and wrist zone intrusion. Ready to start?
