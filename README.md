\# 🚗 Driver Monitoring System

Real-time Driver Monitoring System running on an NVIDIA RTX 4050.



⚡ Performance at a Glance

Metric

CPU

NVIDIA RTX 4050

YOLOv8n inference

~22 FPS

~146 FPS

Latency

~45 ms

~7 ms

Speedup

1×

~6.6×

Full pipeline: ~50–65 FPS at ~15–20 ms/frame.

🧠 What This Project Does

This project implements a multi-modal Driver Monitoring System (DMS) combining classical computer vision, deep learning, temporal reasoning, and GPU acceleration.

The system continuously analyzes:

Camera → Face → Eyes → Head Pose → Gaze → Phone → Driver State → Risk Score

It detects and analyzes:

😴 Drowsiness

👁️ Eye closure and PERCLOS

🥱 Yawning

🧭 Head pose and gaze direction

📱 Phone usage

👤 Driver identity

⚠️ Distraction and vigilance

📊 Real-time driver risk

The goal is not simply to detect individual events, but to build a temporal decision pipeline that combines multiple signals into a stable driver-state estimate.



\*\*Système intelligent de surveillance du conducteur en temps réel\*\* — détection de somnolence, distraction et usage du téléphone via Computer Vision et Deep Learning.



!\[Python](https://img.shields.io/badge/Python-3.10-blue)

!\[PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)

!\[CUDA](https://img.shields.io/badge/CUDA-12.4-green)

!\[YOLOv8](https://img.shields.io/badge/YOLOv8-8.3-orange)

!\[Tests](https://img.shields.io/badge/tests-24%20passed-brightgreen)



\---



\## 📋 Table des matières



\- \[Présentation](#-présentation)

\- \[Fonctionnalités](#-fonctionnalités)

\- \[Architecture](#-architecture)

\- \[Technologies](#-technologies)

\- \[Installation](#-installation)

\- \[Utilisation](#-utilisation)

\- \[Performances](#-performances)

\- \[Tests](#-tests)

\- \[Structure du projet](#-structure-du-projet)

\- \[Algorithmes clés](#-algorithmes-clés)

\- \[Améliorations futures](#-améliorations-futures)

\- \[Licence](#-licence)



\---



\## 🎯 Présentation



Les accidents liés à la \*\*fatigue\*\* et à la \*\*distraction\*\* au volant sont une cause majeure de mortalité routière. Les systèmes DMS (\*Driver Monitoring Systems\*) sont désormais obligatoires dans les nouveaux véhicules en Europe (norme GSR2 depuis 2024).



Ce projet reproduit la logique des DMS industriels :

\- Détection \*\*multi-modale\*\* (visage, yeux, tête, téléphone)

\- \*\*Fusion pondérée\*\* des signaux via un \*Risk Engine\* temporel

\- \*\*Alertes\*\* sonores + visuelles + dashboard web temps réel

\- \*\*Reconnaissance faciale\*\* du conducteur (profils multiples)

\- \*\*Optimisation Edge AI\*\* (export ONNX + benchmark CPU/GPU)



\---



\## ✨ Fonctionnalités



\### 🎥 Vision (perception)

| Module | Description |

|--------|-------------|

| \*\*Face Detection\*\* | 468 landmarks 3D (MediaPipe FaceMesh) |

| \*\*Eye Analysis\*\* | EAR (Eye Aspect Ratio) + calibration auto |

| \*\*Mouth Analysis\*\* | MAR (Mouth Aspect Ratio) — bâillements |

| \*\*Head Pose\*\* | Yaw/Pitch/Roll via `solvePnP` + calibration neutre |

| \*\*Phone Detection\*\* | YOLOv8n deep learning (GPU) |

| \*\*Face Recognition\*\* | Embeddings Facenet (VGGFace2, 512-D) |



\### 🧠 Intelligence (raisonnement)

| Module | Description |

|--------|-------------|

| \*\*Drowsiness\*\* | Durée de fermeture + PERCLOS + bâillements |

| \*\*Distraction\*\* | Regard détourné + téléphone avec seuils temporels |

| \*\*Risk Engine\*\* | Fusion pondérée + lissage EMA (0-100) |

| \*\*Temporal State Engine\*\* | Confirmation + hystérésis (anti-flicker) |

| \*\*Attention Scorer\*\* | Scores attention/vigilance + niveau de risque |



\### 🖥️ Interface \& Monitoring

| Module | Description |

|--------|-------------|

| \*\*Alert Manager\*\* | Alertes sonores (warning/critical) + couleur d'état |

| \*\*HUD temps réel\*\* | Overlay OpenCV : état, scores, FPS, latence, conducteur |

| \*\*SQLite Logger\*\* | Persistance des transitions d'état |

| \*\*Dashboard Streamlit\*\* | Live / Historique / Conducteurs / Rapport PDF |

| \*\*Rapport PDF\*\* | Génération automatique de rapports de session |



\### ⚡ Optimisation

| Module | Description |

|--------|-------------|

| \*\*Export ONNX\*\* | YOLOv8 PyTorch → ONNX (opset 12, simplifié) |

| \*\*Benchmark\*\* | PyTorch vs ONNX × CPU vs GPU |



\---



\## 🏗️ Architecture

┌─────────────────────────────────────────────────────────┐

│ WEBCAM (1280×720) │

└──────────────────────┬──────────────────────────────────┘

│

▼

┌─────────────────────────────────────────────────────────┐

│ VISION LAYER │

│ ┌──────────────┬──────────────┬──────────────────┐ │

│ │ FaceDetector │ PhoneDetector│ FaceRecognizer │ │

│ │ (MediaPipe) │ (YOLOv8n) │ (Facenet VGGFace)│ │

│ └──────┬───────┴──────┬───────┴────────┬─────────┘ │

│ │ │ │ │

│ ▼ ▼ ▼ │

│ ┌──────────────┬──────────────┐ │

│ │ EyeAnalyzer │ HeadPoseEst. │ │

│ │ (EAR/MAR) │ (solvePnP) │ │

│ └──────────────┴──────────────┘ │

└──────────────────────┬──────────────────────────────────┘

│

▼

┌─────────────────────────────────────────────────────────┐

│ INTELLIGENCE LAYER │

│ ┌─────────────────┬──────────────────┐ │

│ │ DrowsinessDet. │ DistractionDet. │ │

│ └────────┬────────┴────────┬─────────┘ │

│ │ │ │

│ ▼ ▼ │

│ ┌────────────────────────────────────┐ │

│ │ RiskEngine (fusion + EMA) │ │

│ └─────────────────┬──────────────────┘ │

│ ▼ │

│ ┌────────────────────────────────────┐ │

│ │ TemporalStateEngine (hystérésis) │ │

│ └─────────────────┬──────────────────┘ │

│ ▼ │

│ ┌────────────────────────────────────┐ │

│ │ AttentionScorer (0-100) │ │

│ └────────────────────────────────────┘ │

└──────────────────────┬──────────────────────────────────┘

│

┌──────────────┼──────────────┐

▼ ▼ ▼

┌───────────────┐ ┌──────────┐ ┌──────────────────┐

│ HUD (OpenCV) │ │ Alerts │ │ SQLite Logger │

│ │ │ (audio) │ │ (events) │

└───────────────┘ └──────────┘ └─────────┬────────┘

│

▼

┌───────────────────────┐

│ Streamlit Dashboard │

│ (Live/Hist/Rapport) │

└───────────────────────┘


\---



\## 🛠️ Technologies



\### Environnement

| Composant | Version | Rôle |

|-----------|---------|------|

| \*\*Python\*\* | 3.10.22 | Langage principal |

| \*\*Anaconda\*\* | env `driver-monitoring` | Environnement |

| \*\*Windows\*\* | 10/11 | OS |



\### Computer Vision

| Librairie | Version | Rôle |

|-----------|---------|------|

| \*\*OpenCV\*\* | 4.11.0 | Capture, traitement, HUD, solvePnP |

| \*\*MediaPipe\*\* | 0.10.14 | 468 landmarks faciaux |

| \*\*NumPy\*\* | 1.26.4 | Calcul matriciel |



\### Deep Learning

| Librairie | Version | Rôle |

|-----------|---------|------|

| \*\*PyTorch\*\* | 2.6.0+cu124 | Framework DL |

| \*\*CUDA\*\* | 12.4 | Accélération GPU |

| \*\*Ultralytics YOLOv8\*\* | 8.3.0 | Détection téléphone |

| \*\*facenet-pytorch\*\* | 2.6.0 | Reconnaissance faciale |

| \*\*ONNX Runtime GPU\*\* | 1.23.2 | Inférence optimisée |



\### Interface \& Data

| Librairie | Version | Rôle |

|-----------|---------|------|

| \*\*Streamlit\*\* | 1.39.0 | Dashboard interactif |

| \*\*Plotly\*\* | 5.24.1 | Graphiques |

| \*\*Pandas\*\* | 2.2.3 | Analyse données |

| \*\*SQLite\*\* | 3.53.4 | Persistance |

| \*\*reportlab\*\* | — | Génération PDF |

| \*\*pygame\*\* | 2.6.1 | Alertes sonores |



\### Matériel recommandé

\- \*\*GPU\*\* : NVIDIA RTX 4050 (ou supérieur) — 6 GB VRAM

\- \*\*Webcam\*\* : 1280×720 @ 30+ FPS

\- \*\*RAM\*\* : 8 GB minimum



\---



\## 🚀 Installation



\### 1. Cloner / créer l'environnement



```bash

conda create -n driver-monitoring python=3.10 -y

conda activate driver-monitoring

