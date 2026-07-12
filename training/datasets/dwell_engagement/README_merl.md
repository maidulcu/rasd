---
annotations_creators: []
language: en
size_categories:
- n<1K
task_categories: []
task_ids: []
pretty_name: merl_shopping
tags:
- fiftyone
- group
dataset_summary: '




  This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 41 samples.


  ## Installation


  If you haven''t already, install FiftyOne:


  ```bash

  pip install -U fiftyone

  ```


  ## Usage


  ```python

  import fiftyone as fo

  from fiftyone.utils.huggingface import load_from_hub


  # Load the dataset

  # Note: other available arguments include ''max_samples'', etc

  dataset = load_from_hub("Voxel51/MERL_Shopping_Dataset")


  # Launch the App

  session = fo.launch_app(dataset)

  ```

  '
---

# Dataset Card for MERL Shopping Dataset

![img/png](merl_dataset.gif)



This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 41 samples.

## Installation

If you haven't already, install FiftyOne:

```bash
pip install -U fiftyone
```

## Usage

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load the dataset
# Note: other available arguments include 'max_samples', etc
dataset = load_from_hub("Voxel51/MERL_Shopping_Dataset")

# Launch the App
session = fo.launch_app(dataset)
```


## Dataset Details

### Dataset Description

The MERL Shopping Dataset is a fine-grained **temporal action detection** dataset consisting of overhead video recordings of people shopping from grocery-store shelving units set up in a lab space. Videos are captured from a single static overhead HD camera. Each subject was recorded across multiple sessions on different days. The dataset is labeled with the start and end frame of every occurrence of 5 fine-grained shopping actions per video.

The dataset was introduced alongside a Multi-Stream Bi-Directional Recurrent Neural Network (MSB-RNN) for fine-grained action detection, presented at CVPR 2016.

- **Curated by:** Bharat Singh (University of Maryland), Tim K. Marks, Michael Jones, Oncel Tuzel (Mitsubishi Electric Research Laboratories), Ming Shao (Northeastern University)
- **Funded by:** Mitsubishi Electric Research Laboratories (MERL)
- **Shared by:** Mitsubishi Electric Research Laboratories (MERL)
- **Language(s) (NLP):** N/A (video dataset)
- **License:** Non-commercial research use only. Copying or reproduction for commercial purposes requires a license from MERL. See `TR2016-080.pdf` for the full copyright notice.

### Dataset Sources

- **Repository:** [MERL Research — Shopping Dataset](http://www.merl.com)
- **Paper:** Singh, B., Marks, T. K., Jones, M., Tuzel, O., & Shao, M. (2016). *A Multi-Stream Bi-Directional Recurrent Neural Network for Fine-Grained Action Detection.* CVPR 2016. MERL Technical Report TR2016-080.

---

## Uses

### Direct Use

- **Temporal action detection:** Training and evaluating models that must localize (start and end frame) every occurrence of each action class within a long untrimmed video.
- **Fine-grained action recognition:** Distinguishing between closely related hand/body actions in a shopping context.
- **Person re-identification across sessions:** The multi-session structure (same subjects across different days) enables studying intra-subject variation.
- **Retail analytics research:** Benchmarking models intended for real-world overhead surveillance in retail environments.

### Out-of-Scope Use

- **Spatial action localization:** Labels are temporal only — no bounding boxes are provided for the actions themselves.
- **Multi-person scenarios:** Each video contains a single actor; the dataset is not suitable for multi-person interaction detection.
- **Commercial deployment** without a license from MERL.
- **General action recognition benchmarks:** The 5 action classes are domain-specific (grocery shopping) and not representative of general human activity.

---

## Dataset Structure

### Overview

| Property | Value |
|---|---|
| Total videos | 106 `.mp4` files |
| Total subjects | 41 |
| Sessions per subject | 1–3 (recorded on different days) |
| Video resolution | 920 × 680 pixels |
| Frame rate | 30 fps |
| Video duration | 116.9s – 205.4s (mean ≈ 138.5s) |
| Camera | Single static overhead HD camera |
| Total action intervals | 5,377 |
| Action classes | 5 |

### File Naming Convention

```
{subject_id}_{session}_crop.mp4        # video
{subject_id}_{session}_label.mat       # labels
```

- `subject_id`: integer 1–41 identifying the person
- `session`: integer 1–3 identifying the recording day for that subject

### Label Format

Each `.mat` file contains a single variable `tlabs` with shape `(5, 1)`. Each of the 5 elements is an `(N, 2)` uint16 array of `[start_frame, end_frame]` pairs (1-indexed, at 30 fps) for that action class.

### Action Classes

| Row | Label | Total Intervals | Mean Duration | Duration Range |
|---|---|---|---|---|
| 0 | Reach to Shelf | 1,711 | ~1.1s (33.5 frames) | 0–173 frames |
| 1 | Retract from Shelf | 1,621 | ~1.2s (36.2 frames) | 7–129 frames |
| 2 | Hand in Shelf | 562 | ~2.1s (62.7 frames) | 5–810 frames |
| 3 | Inspect Product | 674 | ~3.8s (114.6 frames) | 5–748 frames |
| 4 | Inspect Shelf | 809 | ~4.2s (126.6 frames) | 8–1,039 frames |

### Subject / Session Coverage

- **Subjects 1–32:** 3 sessions each → 96 videos
- **Subject 41:** 2 sessions → 2 videos
- **Subjects 33–40:** 1 session each → 8 videos

### FiftyOne Dataset Structure

The dataset is loaded as a **grouped FiftyOne dataset** named `"merl_shopping"` using `load_merl_dataset.py`.

```
fo.Dataset  "merl_shopping"
│
├── group_field: "group"        # one group per subject
│   └── default slice: "session_1"
│
└── Sample fields:
    ├── filepath        (str)   absolute path to .mp4
    ├── group           (Group) FiftyOne group element
    ├── subject_id      (int)   subject identifier (1–41)
    ├── session         (int)   session index (1, 2, or 3)
    └── actions         (TemporalDetections)
            └── detections: list of TemporalDetection
                    ├── label   (str)   action class name
                    └── support ([int, int])  [start_frame, end_frame], 1-indexed
```

**Groups:** 41 (one per subject)
**Slices:** `session_1`, `session_2`, `session_3`

#### Loading the Dataset

```python
import fiftyone as fo

dataset = fo.load_dataset("merl_shopping")

# All sessions for a single subject
subject_view = dataset.select_group_slices().match(
    fo.ViewField("subject_id") == 1
)

# Only session_1 across all subjects
session_1_view = dataset.select_group_slices("session_1")
```

---

## Dataset Creation

### Curation Rationale

The dataset was created to support research in fine-grained temporal action detection in retail/surveillance settings. Existing public benchmarks (e.g., MPII Cooking 2) focused on kitchen activities; this dataset fills the gap for overhead retail scenarios where a static camera, a single actor, and closely related hand/body actions are the norm. The multi-session design per subject supports evaluating intra-subject consistency across days.

### Source Data

#### Data Collection and Processing

Videos were recorded with a static overhead HD camera in a lab space designed to replicate a grocery-store shelving environment. Subjects were asked to shop from the shelving units. Each subject participated in up to 3 separate recording sessions on different days. Videos were then cropped/processed to produce the distributed `.mp4` files (920×680, H.264, 30 fps). Frame-level temporal labels were annotated manually for each of the 5 action classes.

#### Who are the source data producers?

Lab participants recruited by MERL and its collaborating institutions (University of Maryland, Northeastern University). No demographic information about subjects is published.

### Annotations

#### Annotation Process

Annotations consist of `[start_frame, end_frame]` intervals (1-indexed at 30 fps) marking every occurrence of each of the 5 action classes in each video. Labels were produced by researchers at MERL. The evaluation protocol uses the **mid-point hit criterion**: a detection is correct if its temporal midpoint falls within a ground-truth interval.

#### Who are the annotators?

Researchers at Mitsubishi Electric Research Laboratories.

#### Personal and Sensitive Information

Videos contain footage of human subjects recorded in a controlled lab environment. No names, faces (overhead view), or other directly identifying information is visible. Subjects consented to participation in the research study. The dataset is intended for research use only.

---

## Bias, Risks, and Limitations

- **Lab setting:** The shelving environment was constructed in a lab, not a real store. Lighting, background, and shelf layout may not reflect real-world retail diversity.
- **Single camera angle:** All videos use a fixed overhead perspective. Models trained on this data may not generalize to other camera angles or positions.
- **Single actor per video:** No multi-person interaction is present. Models will not learn to handle occlusion between people.
- **Limited subject diversity:** 41 subjects with no published demographic breakdown. The dataset may not represent the full range of body types, heights, or movement styles.
- **Class imbalance:** "Reach to Shelf" and "Retract from Shelf" account for ~61% of all labeled intervals but are the shortest-duration actions. Longer-duration classes like "Inspect Shelf" are underrepresented by count.
- **Domain specificity:** The 5 action classes are narrowly scoped to grocery-shelf shopping and will not transfer directly to other fine-grained action detection tasks.

### Recommendations

Users should be aware that models trained on this dataset are optimized for overhead, single-actor, static-camera settings. Results should not be assumed to generalize to other surveillance or retail environments without further validation. The class imbalance between short reach/retract actions and longer inspection actions should be accounted for during training (e.g., via weighted sampling or per-class evaluation metrics). The non-commercial license must be respected in all use cases.

---

## Citation

**BibTeX:**

```bibtex
@inproceedings{singh2016multistream,
  title     = {A Multi-Stream Bi-Directional Recurrent Neural Network for Fine-Grained Action Detection},
  author    = {Singh, Bharat and Marks, Tim K. and Jones, Michael and Tuzel, Oncel and Shao, Ming},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016},
  note      = {MERL Technical Report TR2016-080}
}
```

**APA:**

Singh, B., Marks, T. K., Jones, M., Tuzel, O., & Shao, M. (2016). A multi-stream bi-directional recurrent neural network for fine-grained action detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. MERL Technical Report TR2016-080.

---

## Dataset Card Contact

For dataset-related questions, contact Mitsubishi Electric Research Laboratories: [http://www.merl.com](http://www.merl.com)
