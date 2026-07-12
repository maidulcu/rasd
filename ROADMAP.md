# Rasd — Product Roadmap

## Open Source (rasd) + Pro (rasd-pro) Strategy

---

## Open Source (rasd) — MIT License

**Goal:** Maximum adoption, community trust, GitHub stars

### Current Features (v1.0)
- YOLOv8 object detection (80 COCO classes)
- ByteTrack person tracking (unique IDs)
- Face detection (Haar cascade)
- Theft detection (unattended objects + concealment)
- Pose estimation (hand-to-pocket, bending)
- Video upload/URL analysis
- FastAPI web interface
- ONNX export support
- CPU-optimized inference

### Monthly Updates (Open Source)

| Month | Feature | Why |
|-------|---------|-----|
| **Month 1** | Webhook support | Let users push alerts to their systems |
| **Month 2** | RTSP stream input | Essential for real CCTV cameras |
| **Month 3** | JSON API responses | Programmatic access to detections |
| **Month 4** | Batch video processing | Analyze multiple videos at once |
| **Month 5** | Docker support | One-command deployment |
| **Month 6** | Custom model training UI | Easy fine-tuning for users |
| **Month 7** | Multi-language docs | Arabic, Spanish, French |
| **Month 8** | GPU acceleration (CUDA) | 10x faster on NVIDIA |
| **Month 9** | Event webhooks | Real-time event streaming |
| **Month 10** | Mobile companion app | View alerts on phone |
| **Month 11** | Plugin system | Community extensions |
| **Month 12** | v2.0 release | Major milestone |

### Open Source Revenue (Indirect)
- Consulting & custom deployment
- Hardware bundles (pre-configured devices)
- Premium support contracts
- Training & workshops

---

## Pro (rasd-pro) — Proprietary

**Goal:** Revenue from businesses needing production features

### Current Features (v1.0)
- Dashboard framework (React)
- Multi-camera manager
- Alert engine (Telegram, Email)
- Face blacklist/whitelist
- Zone editor
- Heatmaps
- API key auth
- Stripe billing
- Docker deployment

### Monthly Updates (Pro)

| Month | Feature | Price Impact |
|-------|---------|--------------|
| **Month 1** | Live camera grid dashboard | Core pro feature |
| **Month 2** | Real-time WebSocket streaming | Dashboard live feed |
| **Month 3** | Advanced analytics (occupancy, flow) | Enterprise tier |
| **Month 4** | POS integration | Retail customers |
| **Month 5** | Cross-camera re-ID | Multi-store tracking |
| **Month 6** | Mobile app (React Native) | Premium add-on |
| **Month 7** | White-label support | Agency partners |
| **Month 8** | AI anomaly detection | Enterprise tier |
| **Month 9** | Compliance reporting (SIRA) | UAE market |
| **Month 10** | Edge device firmware | Hardware bundle |
| **Month 11** | Marketplace for plugins | Platform play |
| **Month 12** | Enterprise SSO/RBAC | Large customers |

---

## Release Schedule

### Weekly (Both Repos)
- Monday: Plan the week
- Tuesday-Thursday: Build features
- Friday: Release + changelog

### Monthly (Both Repos)
- 1st: Release new features
- 15th: Bug fixes + docs
- Last day: Community update

---

## Feature Gating Strategy

```
┌─────────────────────────────────────────────────────────┐
│                    RASD ECOSYSTEM                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐     ┌──────────────────┐          │
│  │   OPEN SOURCE     │     │      PRO         │          │
│  │   (rasd)          │     │   (rasd-pro)     │          │
│  ├──────────────────┤     ├──────────────────┤          │
│  │ ✅ Detection      │     │ ✅ Everything in  │          │
│  │ ✅ Tracking       │     │    open source    │          │
│  │ ✅ Pose           │     │ ✅ Dashboard      │          │
│  │ ✅ Theft detect   │     │ ✅ Multi-cam      │          │
│  │ ✅ Face detect    │     │ ✅ Alerts         │          │
│  │ ✅ Video upload   │     │ ✅ Zones          │          │
│  │ ✅ Basic API      │     │ ✅ Heatmaps       │          │
│  │                   │     │ ✅ Auth + Billing │          │
│  │ ❌ Dashboard      │     │ ✅ Docker         │          │
│  │ ❌ Multi-cam      │     │ ✅ Support        │          │
│  │ ❌ Alerts         │     │                   │          │
│  │ ❌ Zones          │     │                   │          │
│  │ ❌ Heatmaps       │     │                   │          │
│  └──────────────────┘     └──────────────────┘          │
│                                                          │
│  Free forever              $79-199/mo                    │
│  MIT License               Proprietary                   │
│  Community support         Priority support              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Git Workflow

### Open Source (rasd)
```
main (stable) ← releases
  └── dev (development)
       ├── feature/webhooks
       ├── feature/rtsp
       └── fix/tracking-bug
```

### Pro (rasd-pro)
```
main (stable) ← releases
  ├── pro/dev
  │    ├── feature/dashboard
  │    └── feature/alerts
  └── private/enterprise ← customer-specific
```

### Syncing Open Source → Pro
```bash
# Add rasd as submodule in rasd-pro
cd rasd-pro
git submodule add https://github.com/maidulcu/rasd.git

# Update when rasd has new features
cd rasd
git pull origin main
cd ../rasd-pro
git submodule update --remote
```

---

## Documentation Structure

### Open Source Docs (rasd-docs)
```
docs.rasd.ai/
├── /                    # Landing page
├── /getting-started     # Quick start
├── /guides              # Tutorials
├── /api                 # API reference
├── /training            # Custom training
├── /edge                # Edge deployment
└── /community           # Contributing
```

### Pro Docs (rasd-pro/docs)
```
docs.rasd.ai/pro/
├── /dashboard           # Dashboard setup
├── /multi-camera        # Camera management
├── /alerts              # Alert configuration
├── /zones               # Zone editor
├── /billing             # Subscription mgmt
└── /enterprise          # Custom deployment
```

---

## Marketing Calendar

### Monthly Content (Open Source)
| Week | Content | Channel |
|------|---------|---------|
| 1 | New feature announcement | GitHub, X, LinkedIn |
| 2 | Tutorial / How-to | Dev.to, Medium |
| 3 | Community spotlight | X, Discord |
| 4 | Month in review | Newsletter |

### Monthly Content (Pro)
| Week | Content | Channel |
|------|---------|---------|
| 1 | Pro feature update | Email to subscribers |
| 2 | Case study / ROI | LinkedIn, Sales deck |
| 3 | Webinar / Demo | Zoom, YouTube |
| 4 | Customer success | Testimonials |

---

## Success Metrics

### Open Source
| Metric | Month 3 | Month 6 | Month 12 |
|--------|---------|---------|----------|
| GitHub Stars | 500 | 2,000 | 10,000 |
| Contributors | 10 | 25 | 50 |
| Forks | 100 | 400 | 1,000 |
| Issues Closed | 20 | 50 | 100 |

### Pro
| Metric | Month 3 | Month 6 | Month 12 |
|--------|---------|---------|----------|
| Paying Customers | 5 | 20 | 50 |
| MRR | $500 | $2,000 | $5,000 |
| Churn Rate | <10% | <8% | <5% |
| NPS Score | >40 | >50 | >60 |

---

## Immediate Next Steps (This Week)

### Open Source
- [ ] Fix README with demo GIF
- [ ] Add Topics to GitHub repo
- [ ] Post on r/selfhosted
- [ ] Reply to any issues

### Pro
- [ ] Build dashboard skeleton
- [ ] Set up Stripe billing
- [ ] Create landing page
- [ ] First customer outreach

### Training
- [ ] Resume training (5 more epochs)
- [ ] Export best model to ONNX
- [ ] Benchmark on edge device

---

*Last updated: July 2026*
*Owner: Dynamic Web Lab*
