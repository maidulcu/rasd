# Rasd Roadmap

## What Already Exists (We Can Use)

| Feature | Library | Status |
|---------|---------|--------|
| Zone counting (entry/exit) | supervision `PolygonZone`, `LineZone` | ✅ Done |
| Heatmaps | supervision `HeatmapAnnotator` | Can add |
| Dwell time | supervision + tracking | Can add |
| Queue counting | supervision `PolygonZone` | Can add |
| Line crossing | supervision `LineZone` | ✅ Done |
| Object tracking | ultralytics `ByteTrack` | ✅ Done |
| Face detection | OpenCV Haar cascade | ✅ Done |
| Pose estimation | ultralytics YOLOv8-pose | ✅ Done |

## What We Can Add (Free)

### 1. Heatmaps (Easy)
- Use `supervision.HeatmapAnnotator`
- Shows high-traffic areas
- One import away

### 2. Dwell Time (Easy)
- Track how long each person stays in a zone
- Just timestamp tracking + zone detection
- Show average dwell per zone

### 3. Queue Counting (Easy)
- Polygon zone at checkout
- Count people inside = queue length
- Track time = wait time

### 4. Entry/Exit Analytics (Done)
- LineZone already counts in/out
- Just need to expose stats in dashboard

### 5. Conversion Funnel (Medium)
- Entry → Browse → Checkout → Exit
- Track person through zones
- Calculate conversion rate

## What Makes Us Unique

1. **Theft Detection** — Unattended objects + concealment alerts
2. **Pose Estimation** — Hand-in-pocket, bending behavior
3. **Gulf Focus** — Arabic naming, UAE retail
4. **Simple Setup** — Just `pip install`, no Docker/Kafka

## Priority Order

1. Add heatmaps (1 import)
2. Add dwell time tracking
3. Add queue counting
4. Add conversion funnel
5. Expose all stats in dashboard
