"""Driver management: register, list, delete."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path (Streamlit pages don't inherit it)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from vision.face_detection import FaceDetector
from vision.face_recognition import FaceRecognizer

st.set_page_config(page_title="Conducteurs", page_icon="👤", layout="wide")
st.title("👤 Gestion des conducteurs")


# ============================================================
# Cached resources
# ============================================================

@st.cache_resource
def get_recognizer() -> FaceRecognizer:
    return FaceRecognizer()


@st.cache_resource
def get_face_detector() -> FaceDetector:
    return FaceDetector()


recognizer = get_recognizer()
detector = get_face_detector()

# ============================================================
# List existing drivers
# ============================================================

st.subheader("👥 Conducteurs enregistrés")

if len(recognizer.db) == 0:
    st.info("Aucun conducteur enregistré. Ajoutez-en ci-dessous 👇")
else:
    for name in recognizer.db.list_names():
        col1, col2 = st.columns([5, 1])
        col1.markdown(f"**✅ {name}**")
        if col2.button("🗑️ Supprimer", key=f"del_{name}"):
            recognizer.db.remove(name)
            st.success(f"Conducteur '{name}' supprimé.")
            st.rerun()

st.divider()

# ============================================================
# Register new driver
# ============================================================

st.subheader("➕ Enregistrer un nouveau conducteur")

name = st.text_input("Nom du conducteur", placeholder="ex: Ahmed")

col_a, col_b = st.columns([1, 1])

with col_a:
    st.write("**Photo 1 (face avant)**")
    photo1 = st.camera_input("Prenez la photo 1", key="photo1")

with col_b:
    st.write("**Photo 2 (légèrement tournée, optionnel)**")
    photo2 = st.camera_input("Prenez la photo 2", key="photo2")


def process_photo(photo, name_hint: str) -> np.ndarray | None:
    """Convert uploaded photo to embedding, or None."""
    if photo is None:
        return None

    image = Image.open(photo).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    result = detector.process(frame)
    if result is None:
        st.warning(f"❌ Aucun visage détecté dans la photo '{name_hint}'.")
        return None

    x1, y1, x2, y2 = result.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
             caption=f"Visage détecté ({name_hint})")

    emb = recognizer.compute_embedding(frame, result.bbox)
    if emb is None:
        st.warning(f"❌ Embedding échoué pour '{name_hint}'.")
    return emb


if name and (photo1 or photo2):
    st.divider()
    st.write("### Aperçu et validation")

    c1, c2 = st.columns(2)
    with c1:
        emb1 = process_photo(photo1, "photo 1")
    with c2:
        emb2 = process_photo(photo2, "photo 2")

    embeddings = [e for e in [emb1, emb2] if e is not None]

    if not embeddings:
        st.error("Aucun embedding valide. Reprenez les photos.")
    else:
        st.success(f"✅ {len(embeddings)} embedding(s) calculé(s)")

        if st.button("💾 Enregistrer ce conducteur", type="primary"):
            ok = recognizer.register(name, embeddings)
            if ok:
                st.success(f"🎉 Conducteur **{name}** enregistré !")
                st.balloons()
                st.rerun()
            else:
                st.error("Échec de l'enregistrement.")

st.divider()
st.caption(
    "💡 Astuce : prenez 2 photos avec des angles légèrement différents "
    "pour améliorer la robustesse de la reconnaissance."
)