# Introduction-IA
Projet Détection et Classification d'Avions Militaires 
par BARBIER Auréliene et MUSARD Ewen
Bachelor 2
30/11/2026

------------------------------ Partie 1. Analyse des donnees ----------------------------

Ce script Python audite l'ensemble des 1 331 fichiers d'annotations XML (au format Pascal VOC) du jeu train afin de verifier la presence des 20 classes d'avions (A1 a A20) et d'evaluer leur repartition avant l'entrainement. Pour cela, la fonction analyser_annotations_xml parcourt le dossier cible (indique via une chaine brute Windows r"C:\..." pour eviter les erreurs de caracteres d'echappement) et lit chaque fichier XML de maniere securisee grace a ElementTree et un bloc try/except.

Au fil de la lecture, le programme alimente deux compteurs distincts reposant sur Counter : compteur_avions, qui additionne chaque appareil individuel detecte dans les balises <object>/<name>, et compteur_images, qui utilise un ensemble set() par photo afin de ne compter chaque classe qu'une seule fois par image. Cette double approche permet de distinguer la frequence d'apparition d'un modele sur les bases aeriennes de sa tendance a etre stationne en groupe.

Enfin, le script valide en console la presence des 20 classes attendues a travers un tableau trie par frequence decroissante, puis genere un histogramme a barres doubles avec matplotlib (comparant le nombre d'images au total d'avions) pour mettre en evidence le desequilibre du dataset.

-------------------------------------------------------------------------------------------

---------------------------- Partie 2. Preparation des donnees ----------------------------

Ce script Python automatise la creation du dataset d'entrainement en decoupant chaque avion annote dans les images satellites pour le sauvegarder dans un sous-dossier correspondant a sa classe (Vignettes/A1 a Vignettes/A20), structure exigee par Keras. Pour chaque fichier XML, le programme retrouve l'image source (en testant automatiquement plusieurs extensions si necessaire), extrait la zone de l'appareil par tranchage matriciel NumPy (image[ymin:ymax, xmin:xmax]), puis affiche un bilan de repartition par classe via Counter.

Afin de traiter les 1 331 images sans plantage ni creation de vignettes vides, le code s'appuie sur trois securites essentielles :   
Contournement des erreurs Unicode d'OpenCV (lire_image_unicode / ecrire_image_unicode) : Evite les plantages d'OpenCV sous Windows sur les chemins contenant des accents (comme "Introduction a l'IA") en lisant et ecrivant les images sous forme d'octets bruts via NumPy (np.fromfile + cv2.imdecode et cv2.imencode + mode binaire 'wb').
Bornage geometrique des boites : Ramene strictement les coordonnees dans les dimensions reelles de l'image (max(0, ...) et min(...)) et ignore les boites invalides (xmax <= xmin ou ymax <= ymin).
Tolerance aux anomalies (try / except) : Intercepte les erreurs de lecture et comptabilise les images introuvables sans interrompre le traitement global.

--------------------------------------------------------------------------------------------

------------------------------ Partie 3. Classification ------------------------------------

Ce troisieme script Python entraine un reseau de neurones convolutif (CNN) sur le GPU de Google Colab pour classifier les vignettes d'avions selon les 20 categories (A1 a A20) a partir de l'architecture MobileNetV2 (Transfer Learning).

Apres avoir monte Google Drive et extrait l'archive Vignettes.zip, le programme charge les images via image_dataset_from_directory en les redimensionnant automatiquement a 224x224 pixels et en les separant de maniere reproductible (seed=42) en 80 % entrainement et 20 % validation par lots de 32 images (BATCH_SIZE = 32).

Afin de garantir un apprentissage performant, stable et protege contre le sur-apprentissage (overfitting), le code integre quatre mecanismes essentiels :

Reequilibrage des classes (compute_class_weight('balanced')) : Compense le desequilibre du dataset repere en Partie 1 en attribuant un poids plus eleve aux classes d'avions rares lors du calcul de l'erreur, empechant le modele de favoriserseulement les modeles majoritaires.

Fluidification du pipeline GPU (prefetch(AUTOTUNE)) : Prepare les lots d'images suivants en memoire vive pendant que le GPU effectue ses calculs sur le lot actuel, supprimant les temps morts.

Data Augmentation et architecture MobileNetV2 : Applique des transformations aleatoires a la volee (retournements, rotations, zooms, variations de luminosite et de contraste) pour multiplier la variete visuelle, puis passe les images dans MobileNetV2 (pre-entraine sur ImageNet et debloque pour l'affinage avec trainable = True), complete par GlobalAveragePooling2D, BatchNormalization, Dropout(0.4) et une couche finale Dense avec activation softmax.

Pilotes automatiques de l'entrainement (Callbacks + AdamW) : Compile le modele avec l'optimiseur AdamW et la perte sparse_categorical_crossentropy, puis encadre l'apprentissage par trois garde-fous : ReduceLROnPlateau (divise le taux d'apprentissage par 2 si l'erreur de validation stagne), EarlyStopping (arrete l'entrainement des que le modele n'apprend plus et restaure automatiquement ses meilleurs poids via restore_best_weights=True), et ModelCheckpoint (sauvegarde en continu le meilleur fichier modele_avions_propre.keras sur Google Drive).

Enfin, le script exploite l'historique d'entrainement (history) pour tracer avec matplotlib deux graphiques comparatifs (evolution de la Precision / Accuracy et de l'Erreur / Loss sur l'entrainement et la validation) afin de verifier visuellement la convergence du modele.

---------------------------------------------------------------------------------------------
-------------------------------- Partie 4. Detection ----------------------------------------

Ce quatrieme script Python permet de controler visuellement les performances du modele YOLOv8 deja entraine en generant un echantillon de 100 images annotees directement dans le dossier Google Drive Projet_IA_YOLOv8, sans avoir a relancer l'entrainement. Alors que le fichier de soumission resultats_eval_valide.json evalue l'integralite des 2 551 images du jeu eval, cet export visuel est volontairement limite a 100 exemples positifs afin de verifier rapidement la qualite des cadrages sans saturer l'espace Drive ni ralentir l'ecriture des fichiers.   

Apres s'etre connecte a Google Drive, le programme localise automatiquement le fichier de poids best_yolov8_avions.pt grace a une recherche recursive (glob.glob), ce qui lui permet de deduire dynamiquement le chemin du dossier de destination (DOSSIER_YOLO_DRIVE). Il recupere ensuite les images du jeu d'evaluation (eval) : si elles ne sont plus presentes dans le stockage temporaire de Colab (par exemple apres un redemarrage de session), il retrouve automatiquement l'archive eval.zip sur Drive et en extrait uniquement les images valides en ecartant les fichiers caches macOS (__MACOSX et prefixes ._*).

Pour produire rapidement des visuels exploitables tout en evitant les erreurs d'affichage, le script s'appuie sur quatre mecanismes cles :

Inference par lots sur GPU (BATCH_SIZE = 16) : Plutot que de traiter les images une par une, le modele YOLO analyse les photos par paquets de 16 a leur resolution native (imgsz=800), avec un seuil de confiance minimal de 0.25 (conf=0.25) et un filtrage des doublons par suppression des non-maximaux (iou=0.45).

Echantillon cible de 100 images positives (OBJECTIF_VISUELS = 100) : Le code ignore les images sans detection (len(res.boxes) == 0) afin de ne conserver et sauvegarder que les 100 premieres scenes contenant au moins un avion, suffisantes pour le controle visuel et l'illustration du rapport.

Annotation graphique securisee avec OpenCV : Pour chaque boite detectee (box.xyxy), le script convertit l'indice numerique en nom de classe (A1 a A20), trace un cadre vert (cv2.rectangle) et affiche l'etiquette ainsi que le score de confiance en pourcentage (cv2.putText). L'instruction max(15, ymin - 5) garantit que le texte reste toujours lisible a l'interieur de l'image meme lorsqu'un avion se trouve tout en haut du cadre.

Sauvegarde sur Drive et mosaique de controle : Chaque image annotee est enregistree sous le prefixe detect_ directement dans Projet_IA_YOLOv8, et les 12 premiers exemples sont convertis de BGR vers RGB (cv2.cvtColor) pour etre affiches sous forme de mosaique (4 x 3) via matplotlib.