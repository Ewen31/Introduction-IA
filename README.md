# Introduction-IA par BARBIER Aurélien et MUSARD Ewen
Projet Détection et Classification d'Avions Militaires 
par BARBIER Aurélien et MUSARD Ewen
Bachelor 2
30/11/2026

------------------------------------------------------------ Partie 1. Analyse des données ---------------------------------------------------------------

Ce script Python audite l'ensemble des 1 331 fichiers d'annotations XML (au format Pascal VOC) du jeu train afin de vérifier la présence des 20 classes d'avions (A1 à A20) et d'évaluer leur répartition avant l'entraînement. Pour cela, la fonction analyser_annotations_xml parcourt le dossier cible (indiqué via une chaîne brute Windows r"C:\..." pour éviter les erreurs de caractères d'échappement) et lit chaque fichier XML de manière sécurisée grâce à ElementTree et un bloc try/except.

Au fil de la lecture, le programme alimente deux compteurs distincts reposant sur Counter : compteur_avions, qui additionne chaque appareil individuel détecté dans les balises <object>/<name>, et compteur_images, qui utilise un ensemble set() par photo afin de ne compter chaque classe qu'une seule fois par image. Cette double approche permet de distinguer la fréquence d'apparition d'un modèle sur les bases aériennes de sa tendance à être stationné en groupe.

Enfin, le script valide en console la présence des 20 classes attendues à travers un tableau trié par fréquence décroissante, puis génère un histogramme à barres doubles avec matplotlib (comparant le nombre d'images au total d'avions) pour mettre en évidence le déséquilibre du dataset.

----------------------------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------- Partie 2. Préparation des données ------------------------------------------------------------

Ce script Python automatise la création du dataset d'entraînement en découpant chaque avion annoté dans les images satellites pour le sauvegarder dans un sous-dossier correspondant à sa classe (Vignettes/A1 à Vignettes/A20), structure exigée par Keras. Pour chaque fichier XML, le programme retrouve l'image source (en testant automatiquement plusieurs extensions si nécessaire), extrait la zone de l'appareil par tranchage matriciel NumPy (image[ymin:ymax, xmin:xmax]), puis affiche un bilan de répartition par classe via Counter.
Afin de traiter les 1 331 images sans plantage ni création de vignettes vides, le code s'appuie sur trois sécurités essentielles :
Contournement des erreurs Unicode d'OpenCV (lire_image_unicode / ecrire_image_unicode) : Évite les plantages d'OpenCV sous Windows sur les chemins contenant des accents (comme « Introduction à l'IA ») en lisant et écrivant les images sous forme d'octets bruts via NumPy (np.fromfile + cv2.imdecode et cv2.imencode + mode binaire 'wb').
Bornage géométrique des boîtes : Ramène strictement les coordonnées dans les dimensions réelles de l'image (max(0, ...) et min(...)) et ignore les boîtes invalides (xmax <= xmin ou ymax <= ymin).
Tolérance aux anomalies (try / except) : Intercepte les erreurs de lecture et comptabilise les images introuvables sans interrompre le traitement global.

----------------------------------------------------------------------------------------------------------------------------------------------------------

-------------------------------------------------------------- Partie 3. Classification ------------------------------------------------------------------

Ce troisième script Python entraîne un réseau de neurones convolutif (CNN) sur le GPU de Google Colab pour classifier les vignettes d'avions selon les 20 catégories (A1 à A20) à partir de l'architecture MobileNetV2 (Transfer Learning).

Après avoir monté Google Drive et extrait l'archive Vignettes.zip, le programme charge les images via image_dataset_from_directory en les redimensionnant automatiquement à 224x224 pixels et en les séparant de manière reproductible (seed=42) en 80 % entraînement et 20 % validation par lots de 32 images (BATCH_SIZE = 32).

Afin de garantir un apprentissage performant, stable et protégé contre le sur-apprentissage (overfitting), le code intègre quatre mécanismes essentiels :

Rééquilibrage des classes (compute_class_weight('balanced')) : Compense le déséquilibre du dataset repéré en Partie 1 en attribuant un poids plus élevé aux classes d'avions rares lors du calcul de l'erreur, empêchant le modèle de favoriser seulement les modèles majoritaires.

Fluidification du pipeline GPU (prefetch(AUTOTUNE)) : Prépare les lots d'images suivants en mémoire vive pendant que le GPU effectue ses calculs sur le lot actuel, supprimant les temps morts.

Data Augmentation et architecture MobileNetV2 : Applique des transformations aléatoires à la volée (retournements, rotations, zooms, variations de luminosité et de contraste) pour multiplier la variété visuelle, puis passe les images dans MobileNetV2 (pré-entraîné sur ImageNet et débloqué pour l'affinage avec trainable = True), complété par GlobalAveragePooling2D, BatchNormalization, Dropout(0.4) et une couche finale Dense avec activation softmax.

Pilotes automatiques de l'entraînement (Callbacks + AdamW) : Compile le modèle avec l'optimiseur AdamW et la perte sparse_categorical_crossentropy, puis encadre l'apprentissage par trois garde-fous : ReduceLROnPlateau (divise le taux d'apprentissage par 2 si l'erreur de validation stagne), EarlyStopping (arrête l'entraînement dès que le modèle n'apprend plus et restaure automatiquement ses meilleurs poids via restore_best_weights=True), et ModelCheckpoint (sauvegarde en continu le meilleur fichier modele_avions_propre.keras sur Google Drive).

Enfin, le script exploite l'historique d'entraînement (history) pour tracer avec matplotlib deux graphiques comparatifs (évolution de la Précision / Accuracy et de l'Erreur / Loss sur l'entraînement et la validation) afin de vérifier visuellement la convergence du modèle.

-----------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------ Partie 4. Détection --------------------------------------------------------------------

Ce script intermédiaire de la Partie 4 permet de contrôler visuellement les performances du modèle YOLOv8 déjà entraîné en générant un échantillon de 100 images annotées directement dans le dossier Google Drive Projet_IA_YOLOv8, sans avoir à relancer l'entraînement. Alors que le script d'inférence finale (resultats_eval_soumission.json) évalue l'intégralité des 2 511 images inédites du jeu eval en combinant YOLOv8 et MobileNetV2, cet export visuel est volontairement limité à 100 exemples positifs issus du jeu d'entraînement (train) afin de vérifier rapidement la qualité des cadrages à l'œil nu et d'illustrer le rapport sans saturer l'espace Drive.

Après s'être connecté à Google Drive, le programme localise automatiquement le fichier de poids best_yolov8_avions.pt grâce à une recherche récursive (glob.glob), ce qui lui permet de déduire dynamiquement le chemin du dossier de destination (DOSSIER_YOLO_DRIVE). Il récupère ensuite les images du jeu d'entraînement (train) : si elles ne sont plus présentes dans le stockage temporaire de Colab (par exemple après un redémarrage de session), il retrouve automatiquement l'archive train.zip sur Drive et en extrait uniquement les images valides en écartant les fichiers cachés macOS (__MACOSX et préfixes ._*).

Pour produire rapidement des visuels exploitables tout en évitant les erreurs d'affichage, le script s'appuie sur quatre mécanismes clés :

Inférence par lots sur GPU (BATCH_SIZE = 16) : Plutôt que de traiter les images une par une, le modèle YOLO analyse les photos par paquets de 16 à leur resolution native (imgsz=800), avec un seuil de confiance minimal de 0,25 (conf=0.25) et un filtrage des doublons par suppression des non-maximaux (iou=0.45).

Échantillon cible de 100 images positives (OBJECTIF_VISUELS = 100) : Le code ignore les images sans détection (len(res.boxes) == 0) afin de ne conserver et sauvegarder que les 100 premières scènes contenant au moins un avion.

Annotation graphique sécurisée avec OpenCV : Pour chaque boîte détectée (box.xyxy), le script convertit l'indice numérique en nom de classe (A1 à A20), trace un cadre vert (cv2.rectangle) et affiche l'étiquette ainsi que le score de confiance en pourcentage (cv2.putText). L'instruction max(15, ymin - 5) garantit que le texte reste toujours lisible à l'intérieur de l'image même lorsqu'un avion se trouve tout en haut du cadre.

Sauvegarde sur Drive et mosaïque de contrôle : Chaque image annotée est enregistrée sous le préfixe detect_train_ directement dans Projet_IA_YOLOv8, et les 12 premiers exemples sont convertis de BGR vers RGB (cv2.cvtColor) pour être affichés sous forme de mosaïque (4 × 3) via matplotlib.

----------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------- Lien entre classification et détection ----------------------------------------------------------

Ce dernier script constitue l'étape finale du projet : il combine le détecteur YOLOv8 (chargé de localiser les avions sur les images satellites complètes en 800x800 pixels) et le classifieur MobileNetV2 (entraîné en Partie 3 sur des vignettes en 224x224 pixels) afin d'analyser les 2 511 images inédites du jeu d'évaluation (eval) et de générer le fichier officiel de soumission resultats_eval_soumission.json.

Le script se connecte à Google Drive pour charger le modèle de classification MobileNetV2 (modele_avions_propre.keras). Il met également en place une table de correspondance entre l'ordre numérique naturel utilisé par YOLOv8 (A1, A2, ..., A20) et l'ordre alphabétique strict utilisé par Keras lors de la lecture des sous-dossiers (A1, A10, A11, ..., A2, A20), évitant ainsi tout décalage d'étiquette lors de la fusion des résultats.

Le script vérifie si les poids du détecteur YOLOv8 (best_yolov8_avions.pt) existent déjà sur Google Drive. S'ils sont présents, ils sont chargés immédiatement. Dans le cas contraire, le programme décompresse automatiquement l'archive d'entraînement (train) en filtrant les fichiers cachés macOS (__MACOSX et ._*), convertit les annotations Pascal VOC (.xml) au format normalisé YOLO (.txt), sépare les données en 80 % entraînement et 20 % validation, puis entraîne un modèle yolov8s.pt en résolution native 800x800 pixels avant de sauvegarder les poids sur Drive.

Le script extrait proprement les 2 511 images satellites du jeu d'évaluation eval (dépourvu d'annotations XML). Les deux réseaux s'exécutent ensuite avec leurs poids figés, sans aucun ré-entraînement ni ajustement de seuil sur ce jeu, garantissant une évaluation objective sans fuite de données.

Les images sont traitées par lots de 16 (batch size) selon un pipeline en deux temps :
Localisation avec YOLOv8 : le détecteur repère chaque avion avec un seuil de confiance minimal de 0,25 et supprime les boîtes redondantes qui se chevauchent grâce à l'algorithme NMS (IoU = 0,45).
Classification fine avec MobileNetV2 : chaque zone détectée est découpée dans l'image d'origine, convertie de BGR vers RGB, redimensionnée en 224x224 pixels et transmise au réseau MobileNetV2.
Règle de fusion : lorsque les deux modèles prédisent la même classe, celle-ci est retenue avec la moyenne de leurs scores de confiance. En cas de désaccord, la priorité est donnée à YOLOv8 pour les très petits appareils (moins de 35 pixels de côté, où le contexte global de la piste est déterminant) ou au modèle affichant le score de confiance le plus élevé.

Avant de finaliser le fichier, le programme vérifie que les 2 511 images sont bien répertoriées (y compris les images sans avion sous forme de liste vide). Le fichier resultats_eval_soumission.json est ensuite sauvegardé sur Google Drive et téléchargé automatiquement.









