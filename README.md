# VGGFace2 ArcFace Research

Research repository for the **Missing Person Finder** face-recognition research.

## Objective

Evaluate an ArcFace-based face recognition pipeline, identify difficult recognition cases, and test controlled preprocessing/alignment improvements.

## Dataset

VGGFace2 training data was used to create a controlled research subset:

- Identities: 8,631
- Images per identity: 15
- Total images: 129,465
- Random seed: 42
- Unreadable images: 0

The selected-image manifest is:

`metadata/vggface2_selected_129465.csv`

The full image dataset is not stored in this repository.

## Pipeline

```text
VGGFace2 Image
      ↓
Bounding-box crop
      ↓
Five-point landmark alignment
      ↓
112 × 112 face
      ↓
ArcFace
      ↓
512-D embedding
      ↓
Cosine similarity
      ↓
Genuine / Impostor evaluation
