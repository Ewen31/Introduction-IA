import os
import glob
import zipfile
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from google.colab import drive

# ==============================================================================
# 1. CONNEXION DRIVE & LOCALISATION DU DOSSIER Projet_IA_YOLOv8
# ==============================================================================
if not os.path.exists("/content/drive/MyDrive"):
    drive.mount("/content/drive")

weights_list = glob.glob("/content/drive/MyDrive/**/Projet_IA_YOLOv8/best_yolov8_avions.pt", recursive=True)
if not weights_list:
    weights_list = glob.glob("/content/drive/MyDrive/**/best_yolov8_avions.pt", recursive=True)
if not weights_list:
    raise FileNotFoundError("❌ Fichier best_yolov8_avions.pt introuvable sur Google Drive.")

BEST_WEIGHTS = weights_list[0]
DOSSIER_YOLO_DRIVE = os.path.dirname(BEST_WEIGHTS)
print(f"📂 Dossier cible Google Drive : {DOSSIER_YOLO_DRIVE}")

# ==============================================================================
# 2. RÉCUPÉRATION DES IMAGES D'ÉVALUATION (EVAL)
# ==============================================================================
extensions_valides = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
eval_images = []

for dossier_candidat in ["/content/yolo_aircraft/raw_eval", "/content/raw_eval"]:
    if os.path.exists(dossier_candidat):
        for root, _, files in os.walk(dossier_candidat):
            if "__MACOSX" in root:
                continue
            eval_images.extend([
                os.path.join(root, f) for f in files
                if f.lower().endswith(extensions_valides) and not f.startswith("._")
            ])
        if eval_images:
            break

if not eval_images:
    print("⚡ Extraction des images depuis eval.zip sur Drive...")
    eval_zips = glob.glob("/content/drive/MyDrive/**/*eval*.zip", recursive=True)
    if not eval_zips:
        raise FileNotFoundError("❌ Archive eval.zip introuvable sur Google Drive.")
    raw_eval_dir = "/content/yolo_aircraft/raw_eval"
    os.makedirs(raw_eval_dir, exist_ok=True)
    with zipfile.ZipFile(eval_zips[0], "r") as zf:
        for member in zf.infolist():
            fname = os.path.basename(member.filename)
            if "__MACOSX" in member.filename or fname.startswith("._") or member.is_dir():
                continue
            if fname.lower().endswith(extensions_valides):
                dest_path = os.path.join(raw_eval_dir, fname)
                with zf.open(member) as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                eval_images.append(dest_path)

eval_images = sorted(eval_images)
print(f"🖼️ {len(eval_images)} images d'évaluation prêtes.")

# ==============================================================================
# 3. GÉNÉRATION DE 100 PHOTOS ANNOTÉES DIRECTEMENT DANS Projet_IA_YOLOv8
# ==============================================================================
CLASSES = [f"A{i}" for i in range(1, 21)]
model = YOLO(BEST_WEIGHTS)

OBJECTIF_VISUELS = 100
visuels_sauvegardes = 0
images_apercu = []
BATCH_SIZE = 16

print(f"\n🚀 Génération de {OBJECTIF_VISUELS} images avec avions détectés dans {DOSSIER_YOLO_DRIVE}...")

for i in range(0, len(eval_images), BATCH_SIZE):
    if visuels_sauvegardes >= OBJECTIF_VISUELS:
        break

    lot_chemins = eval_images[i : i + BATCH_SIZE]
    resultats = model.predict(source=lot_chemins, imgsz=800, conf=0.25, iou=0.45, verbose=False)

    for img_path, res in zip(lot_chemins, resultats):
        if visuels_sauvegardes >= OBJECTIF_VISUELS:
            break
        if len(res.boxes) == 0:
            continue  # On ne garde que les photos où au moins 1 avion est détecté

        img_visuel = cv2.imread(img_path)
        if img_visuel is None:
            continue

        for box in res.boxes:
            xmin, ymin, xmax, ymax = [int(round(v)) for v in box.xyxy[0].tolist()]
            cls_name = CLASSES[int(box.cls[0].item())]
            score = float(box.conf[0].item())

            cv2.rectangle(img_visuel, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                img_visuel,
                f"{cls_name} {score*100:.0f}%",
                (xmin, max(15, ymin - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        nom_fichier = os.path.basename(img_path)
        chemin_sortie = os.path.join(DOSSIER_YOLO_DRIVE, f"detect_{nom_fichier}")
        cv2.imwrite(chemin_sortie, img_visuel)
        visuels_sauvegardes += 1

        if len(images_apercu) < 12:
            images_apercu.append((nom_fichier, cv2.cvtColor(img_visuel, cv2.COLOR_BGR2RGB)))

print(f"\n🎉 Terminé ! {visuels_sauvegardes} photos enregistrées directement dans : {DOSSIER_YOLO_DRIVE}")

# Affichage d'une mosaïque de 12 photos directement sous la cellule
if images_apercu:
    nb_show = len(images_apercu)
    cols = 4
    rows = (nb_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
    axes = np.array(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        if idx < nb_show:
            nom, img_rgb = images_apercu[idx]
            ax.imshow(img_rgb)
            ax.set_title(nom)
        ax.axis("off")
    plt.tight_layout()
    plt.show()