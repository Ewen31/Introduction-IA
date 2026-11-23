import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analyser_annotations_xml(dossier_annotations):
    """
    Parcourt un dossier contenant des fichiers XML (format Pascal VOC).
    Extrait les classes, compte les occurrences totales ET le nombre d'images par classe.
    """
    compteur_avions = Counter()      # Compte chaque avion individuellement
    compteur_images = Counter()      # Compte les images contenant au moins un avion de la classe
    total_fichiers_xml = 0
    fichiers_sans_objets = 0

    print(f"Analyse en cours du dossier : {dossier_annotations}...\n")

    for nom_fichier in os.listdir(dossier_annotations):
        if nom_fichier.lower().endswith('.xml'):
            total_fichiers_xml += 1
            chemin_complet = os.path.join(dossier_annotations, nom_fichier)
            
            try:
                arbre = ET.parse(chemin_complet)
                racine = arbre.getroot()
                
                objets = racine.findall('object')
                if not objets:
                    fichiers_sans_objets += 1
                    continue
                    
                # Utiliser un set pour savoir quelles classes sont présentes dans CETTE image spécifiquement
                classes_dans_image = set()
                
                for objet in objets:
                    balise_nom = objet.find('name')
                    if balise_nom is not None and balise_nom.text:
                        nom_classe = balise_nom.text.strip()
                        compteur_avions[nom_classe] += 1
                        classes_dans_image.add(nom_classe)
                
                # Ajouter +1 au compteur d'images pour chaque classe trouvée dans cette image
                for classe in classes_dans_image:
                    compteur_images[classe] += 1
                        
            except ET.ParseError:
                print(f"Erreur de lecture sur le fichier : {nom_fichier}")

    nombre_classes = len(compteur_images)
    
    # --- Affichage des résultats en console ---
    print("=== RÉSULTATS DE L'ANALYSE DES ANNOTATIONS ===")
    print(f"Total des images (fichiers XML) : {total_fichiers_xml}")
    print(f"Total des classes détectées     : {nombre_classes}")
    
    if fichiers_sans_objets > 0:
        print(f"Attention : {fichiers_sans_objets} image(s) ne contiennent aucun avion annoté.")
    print("-" * 55)

    if nombre_classes == 20:
        print("✅ SUCCÈS : Les 20 classes sont bien présentes !")
    else:
        print(f"⚠️ ATTENTION : Attendu: 20, Trouvé: {nombre_classes}.")
    
    print("-" * 55)
    print(f"{'MODÈLE':<15} | {'NB IMAGES':<15} | {'NB TOTAL AVIONS'}")
    print("-" * 55)
    # On trie par le nombre d'images décroissant
    for classe, nb_img in compteur_images.most_common():
        nb_av = compteur_avions[classe]
        print(f" - {classe:<12} | {nb_img:<15} | {nb_av}")

    # --- Affichage graphique (Double Barres) ---
    if nombre_classes > 0:
        classes = [item[0] for item in compteur_images.most_common()]
        valeurs_images = [compteur_images[c] for c in classes]
        valeurs_avions = [compteur_avions[c] for c in classes]
        
        x = np.arange(len(classes))
        largeur_barre = 0.35
        
        plt.figure(figsize=(14, 7))
        plt.bar(x - largeur_barre/2, valeurs_images, largeur_barre, label='Nombre d\'images', color='skyblue', edgecolor='black')
        plt.bar(x + largeur_barre/2, valeurs_avions, largeur_barre, label='Nombre total d\'avions', color='coral', edgecolor='black')
        
        plt.title("Répartition des modèles d'avions (Images vs Nombre total)", fontsize=14)
        plt.xlabel("Modèles (Classes)", fontsize=12)
        plt.ylabel("Quantité", fontsize=12)
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# --- EXÉCUTION ---
chemin_annotations = r"C:\Users\ewenm\OneDrive\IPSA\B2\Introduction à l'IA\military-aircraft-recognition-dataset.train\student-dataset 2\annotations"
analyser_annotations_xml(chemin_annotations)