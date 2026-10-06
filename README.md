# Driver Monitoring System

Système de surveillance de conducteur en temps réel — détection de somnolence, distraction, et usage du téléphone à partir d'une simple webcam.

Tourne à **~50 FPS en 720p sur une RTX 4050 laptop (6 GB)**, avec phone detection + face recognition actifs simultanément. Stable sur des sessions de 30 minutes.

## Pourquoi ce projet

Je voulais apprendre PyTorch, CUDA et YOLO sur un projet concret qui ait du sens. La fatigue au volant tue — et les DMS (Driver Monitoring Systems) sont devenus obligatoires dans les véhicules neufs en Europe depuis 2024 (norme GSR2). C'était l'occasion de toucher à la fois à la vision par ordinateur, au deep learning temps réel, et à l'optimisation GPU sur quelque chose de tangible.

## Ce que ça fait

Pipeline complet, du pixel brut jusqu'au score de risque :
Webcam → Face → Eyes → Head → Gaze → Phone → Driver State → Risk

# Driver Monitoring System

Real-time driver monitoring from a single webcam — drowsiness, distraction, and phone usage detection.

Runs at **~50 FPS at 720p on an RTX 4050 laptop (6 GB)** with phone detection and face recognition both active. Stable over 30-minute sessions.

## Why I built this

I wanted to learn PyTorch, CUDA, and YOLO on something concrete. Driver fatigue kills people — and Driver Monitoring Systems have been mandatory in new cars in Europe since 2024 (GSR2 regulation). It was a good excuse to touch computer vision, real-time deep learning, and GPU optimization on a project that matters.

## What it does

Full pipeline, from raw pixels to a risk score:
conda create -n driver-monitoring python=3.10 -y
conda activate driver-monitoring
pip install -r requirements.txt


## Usage
Main pipeline
python main.py

Streamlit dashboard (live + history + reports)
streamlit run dashboard/app.py

PyTorch vs ONNX × CPU vs GPU benchmark
python benchmark.py


## Limitations

- **No testing on real drivers in real driving conditions.** All measurements are webcam-at-desk.
- **No mobile export yet.** The ONNX model isn't quantized — int8 would be needed to run on a Jetson Nano or Raspberry Pi.
- **Face recognition is sensitive to lighting.** Backlight or night driving makes Facenet embeddings unstable.
- **The Risk Engine weights are hand-tuned.** Fusion weights (EAR, MAR, head, phone) are fixed. A production DMS would learn these per driver.

## What's next

Three directions:

1. **int8 export** to port the system to Jetson Nano — a real embedded DMS.
2. **Fine-tune the Risk Engine** on real sessions with user feedback.
3. **Add seatbelt + hands-on-wheel detection** — two signals required by the GSR2 regulation.

## Context

Personal project, built alongside my studies at ENIG (École Nationale d'Ingénieurs de Gabès) to go deeper into real-time deep learning and GPU optimization. Goal: build a strong portfolio for international internships in HPC / computer vision.

Contact: tayari.bahaeddine@gmail.com
