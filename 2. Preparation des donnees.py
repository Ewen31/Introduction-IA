import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from collections import Counter

def lire_image_unicode(chemin_image):
    """Lecture d'image compatible avec les chemins contenant des accents/caractères spéciaux."""
    image_bytes = np.fromfile(chemin_image, dtype=np.uint8)
    return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

def ecrire_image_unicode(chemin_sauvegarde, image):
    """Écriture d'image compatible avec les chemins contenant des accents/caractères spéciaux."""
    extension = os.path.splitext(chemin_sauvegarde)[1]
    succes, buffer = cv2.imencode(extension, image)
    if succes:
        with open(chemin_sauvegarde, 'wb') as f:
            f.write(buffer)

def generer_vignettes_dans_dossier_vignettes(dossier_images, dossier_annotations, dossier_vignettes):
    """
    Extrait une vignette (crop) pour chaque avion annoté dans les fichiers XML
    et les classe dans le dossier 'Vignettes' séparé par des sous-dossiers de classes.
    """
    os.makedirs(dossier_vignettes, exist_ok=True)
    fichiers_xml = [f for f in os.listdir(dossier_annotations) if f.lower().endswith('.xml')]
    
    print(f"=== GÉNÉRATION DES VIGNETTES DANS LE DOSSIER 'Vignettes' ===")
    print(f"Traitement de {len(fichiers_xml)} fichiers XML d'annotations...\n")
    
    compteur_vignettes = Counter()
    images_non_trouvees = 0
    erreurs = 0

    for fichier_xml in fichiers_xml:
        chemin_xml = os.path.join(dossier_annotations, fichier_xml)
        try:
            arbre = ET.parse(chemin_xml)
            racine = arbre.getroot()
            
            # Recherche du nom de l'image source
            balise_filename = racine.find('filename')
            nom_image = balise_filename.text if (balise_filename is not None and balise_filename.text) else fichier_xml.replace('.xml', '.jpg')
            chemin_image = os.path.join(dossier_images, nom_image)
            
            # Résolution de l'extension de fichier si besoin
            if not os.path.exists(chemin_image):
                base_nom = os.path.splitext(nom_image)[0]
                trouve = False
                for ext in ['.jpg', '.png', '.jpeg', '.tif', '.JPG', '.PNG']:
                    chemin_test = os.path.join(dossier_images, base_nom + ext)
                    if os.path.exists(chemin_test):
                        chemin_image = chemin_test
                        trouve = True
                        break
                if not trouve:
                    images_non_trouvees += 1
                    continue

            # Chargement de l'image
            image = lire_image_unicode(chemin_image)
            if image is None:
                erreurs += 1
                continue

            hauteur_img, largeur_img = image.shape[:2]

            # Extraction de chaque avion dans l'image
            objets = racine.findall('object')
            for index_objet, objet in enumerate(objets):
                nom_classe = objet.find('name').text.strip()
                bndbox = objet.find('bndbox')
                
                # Coordonnées du crop
                xmin = max(0, int(float(bndbox.find('xmin').text)))
                ymin = max(0, int(float(bndbox.find('ymin').text)))
                xmax = min(largeur_img, int(float(bndbox.find('xmax').text)))
                ymax = min(hauteur_img, int(float(bndbox.find('ymax').text)))
                
                if xmax <= xmin or ymax <= ymin:
                    continue

                # Découpage de la vignette
                vignette_avion = image[ymin:ymax, xmin:xmax]
                
                # Création du sous-dossier de la classe DANS 'Vignettes'
                dossier_classe = os.path.join(dossier_vignettes, nom_classe)
                os.makedirs(dossier_classe, exist_ok=True)
                
                # Nommage et sauvegarde
                base_nom_img = os.path.splitext(os.path.basename(chemin_image))[0]
                nom_fichier_crop = f"{base_nom_img}_avion_{index_objet + 1}.png"
                chemin_sauvegarde = os.path.join(dossier_classe, nom_fichier_crop)
                
                ecrire_image_unicode(chemin_sauvegarde, vignette_avion)
                compteur_vignettes[nom_classe] += 1
                
        except Exception as e:
            erreurs += 1

    # --- BILAN DES RÉSULTATS ---
    total_crops = sum(compteur_vignettes.values())
    print("=== BILAN DU CLASSEMENT DANS 'Vignettes' ===")
    print(f"Chemin du dossier principal         : {dossier_vignettes}")
    print(f"Total des vignettes générées        : {total_crops}")
    print(f"Nombre de sous-dossiers créés       : {len(compteur_vignettes)}")
    if images_non_trouvees > 0:
        print(f"⚠️ Images sources introuvables      : {images_non_trouvees}")
    if erreurs > 0:
        print(f"⚠️ Fichiers d'annotation en erreur  : {erreurs}")
    print("-" * 50)
    print("Répartition des vignettes enregistrées :")
    for classe, nombre in compteur_vignettes.most_common():
        print(f" - Vignettes/{classe:<15} : {nombre} images")

# --- CONFIGURATION ET EXÉCUTION ---
chemin_images = r"C:\Users\ewenm\OneDrive\IPSA\B2\Introduction à l'IA\military-aircraft-recognition-dataset.train\student-dataset 2\images"
chemin_annotations = r"C:\Users\ewenm\OneDrive\IPSA\B2\Introduction à l'IA\military-aircraft-recognition-dataset.train\student-dataset 2\annotations"

# Destination dans le dossier "Vignettes"
chemin_sortie_vignettes = r"C:\Users\ewenm\OneDrive\IPSA\B2\Introduction à l'IA\military-aircraft-recognition-dataset.train\student-dataset 2\Vignettes"

generer_vignettes_dans_dossier_vignettes(chemin_images, chemin_annotations, chemin_sortie_vignettes)