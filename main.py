import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinterdnd2 import DND_FILES, TkinterDnD
import webbrowser
import cv2
import numpy as np
import easyocr
import threading
from PIL import Image, ImageTk, ImageSequence
from tkinter import ttk
import os

# Variables globales pour les textes en différentes langues
lang_texts = {
    'en': {
        'title': "CarsDetect",
        'load_video': "Load Video",
        'drop_video': "Drag and drop your video here",
        'encadrer_plaques': "Draw boxes around plates",
        'afficher_numeros': "Display plate numbers",
        'encadrer_voitures': "Draw boxes around cars",
        'processing_done': "Processing completed and video saved.",
        'processing_error': "Error: Unable to open the video",
        'file_not_exist': "The file does not exist.",
        'info': "Info",
        'settings': "Settings",
        'video': "Video",
        'select_language': "Select your language:",
        'select_language_title': "Select Language",
        'enter_filename': "Enter the output file name (without extension):",
        'warning_filename': "You must enter a file name to continue."
    },
    'fr': {
        'title': "CarsDetect",
        'load_video': "Charger une Vidéo",
        'drop_video': "Glissez-déposez votre vidéo ici",
        'encadrer_plaques': "Encadrer les plaques",
        'afficher_numeros': "Afficher les numéros des plaques",
        'encadrer_voitures': "Encadrer les voitures",
        'processing_done': "Traitement terminé et vidéo enregistrée.",
        'processing_error': "Erreur : Impossible d'ouvrir la vidéo",
        'file_not_exist': "Le fichier n'existe pas.",
        'info': "Info",
        'settings': "Paramètres",
        'video': "Vidéo",
        'select_language': "Sélectionnez votre langue :",
        'select_language_title': "Sélection de la Langue",
        'enter_filename': "Entrez le nom du fichier de sortie (sans extension) :",
        'warning_filename': "Vous devez entrer un nom de fichier pour continuer."
    }
}

# Fonction pour mettre à jour les textes de l'interface en fonction de la langue
def update_texts(lang):
    global texts
    texts = lang_texts[lang]

def process_video(video_path, output_filename):
    # Désactiver le bouton et masquer la zone de glisser-déposer
    btn_load.config(state=tk.DISABLED)
    drop_label.pack_forget()
    gif_label.pack_forget()

    # Chemins d'accès aux fichiers SSD
    prototxt_path = "model/deploy.prototxt"
    model_path = "model/mobilenet_iter_73000.caffemodel"

    # Vérifier si les fichiers de modèle existent
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        messagebox.showerror("Erreur", "Les fichiers de modèle ne sont pas trouvés.")
        btn_load.config(state=tk.NORMAL)
        drop_label.pack(pady=20)
        gif_label.pack(pady=10)
        return

    # Charger le modèle SSD
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Initialiser EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Erreur", texts['processing_error'])
        btn_load.config(state=tk.NORMAL)
        drop_label.pack(pady=20)
        gif_label.pack(pady=10)
        return

    # Obtenir le nombre total de frames dans la vidéo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # Définir le codec et créer un objet VideoWriter pour enregistrer la vidéo
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                if class_id == 7:  # ID pour 'car' dans le modèle COCO
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (x, y, x1, y1) = box.astype("int")

                    # Vérifiez que les coordonnées sont dans les limites de l'image
                    x = max(0, x)
                    y = max(0, y)
                    x1 = min(width, x1)
                    y1 = min(height, y1)

                    if x < x1 and y < y1:  # Vérifier si la ROI est valide
                        # Dessiner le rectangle rouge autour de la voiture si l'option est activée
                        if draw_cars.get():
                            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

                        # Découper la région de la plaque d'immatriculation (supposée)
                        roi = frame[y:y1, x:x1]

                        # Appliquer les techniques de prétraitement pour OCR
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        gray_roi = cv2.bilateralFilter(gray_roi, 11, 17, 17)
                        edged_roi = cv2.Canny(gray_roi, 30, 200)

                        try:
                            # Utiliser OCR pour lire le texte dans la ROI prétraitée
                            plate_texts = reader.readtext(edged_roi)

                            for plate in plate_texts:
                                bbox, text, conf = plate
                                if conf > 0.5:
                                    # Les coordonnées de la bbox sont relatives à la ROI
                                    (plate_x, plate_y), (plate_x1, plate_y1) = bbox[0], bbox[2]
                                    plate_x = int(plate_x + x)
                                    plate_y = int(plate_y + y)
                                    plate_x1 = int(plate_x1 + x)
                                    plate_y1 = int(plate_y1 + y)

                                    # Dessiner les rectangles autour des plaques d'immatriculation
                                    if draw_boxes.get():
                                        cv2.rectangle(frame, (plate_x, plate_y), (plate_x1, plate_y1), (0, 255, 0), 2)

                                    # Afficher les numéros de plaque d'immatriculation
                                    if show_plate_text.get():
                                        cv2.putText(frame, text, (plate_x, plate_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        except Exception as e:
                            print(f"Erreur de lecture OCR : {e}")

        out.write(frame)

        # Afficher le frame dans la GUI
        show_frame(frame)

        # Mettre à jour la barre de progression
        current_frame += 1
        update_progress(current_frame, total_frames)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    messagebox.showinfo(texts['info'], texts['processing_done'])

    # Réactiver le bouton et afficher la zone de glisser-déposer
    btn_load.config(state=tk.NORMAL)
    drop_label.pack(pady=20)
    gif_label.pack(pady=10)

# Fonction pour charger la vidéo
def load_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
    if video_path:
        # Utiliser les textes traduits pour la boîte de dialogue de nom de fichier
        output_filename = simpledialog.askstring(
            texts['title'], 
            texts['enter_filename']
        )
        if output_filename:
            output_filename = output_filename + ".avi"
            lbl_video.pack(pady=10)
            threading.Thread(target=process_video, args=(video_path, output_filename)).start()
        else:
            messagebox.showwarning("Avertissement", texts['warning_filename'])

def show_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    
    lbl_video.config(image=img)
    lbl_video.image = img

# Fonction pour gérer le glisser-déposer
def drop(event):
    video_path = event.data.strip("{}")
    if os.path.isfile(video_path):
        # Demander à l'utilisateur de saisir le nom du fichier de sortie
        output_filename = simpledialog.askstring("Nom du fichier de sortie", "Entrez le nom du fichier de sortie (sans extension):")
        if output_filename:
            # Ajouter l'extension .avi
            output_filename = output_filename + ".avi"

            # Afficher le label de la vidéo
            lbl_video.pack(pady=10)
            threading.Thread(target=process_video, args=(video_path, output_filename)).start()
        else:
            messagebox.showwarning("Avertissement", "Vous devez entrer un nom de fichier pour continuer.")
    else:
        messagebox.showerror("Erreur", texts['file_not_exist'])

# Fonction pour mettre à jour la barre de progression
def update_progress(current_frame, total_frames):
    progress = (current_frame / total_frames) * 100
    progress_var.set(progress)

# Fonction pour sélectionner la langue
def select_language(lang):
    update_texts(lang)
    lang_selection.destroy()
    setup_gui()

# Configuration initiale de sélection de la langue
lang_selection = tk.Tk()
lang_selection.title(lang_texts['en']['select_language_title'])
lang_selection.geometry("300x150")
lang_selection.resizable(False, False)

label_select_lang = tk.Label(lang_selection, text=lang_texts['en']['select_language'], font=("Arial", 12))
label_select_lang.pack(pady=10)

btn_english = tk.Button(lang_selection, text="English", width=15, command=lambda: select_language('en'))
btn_english.pack(pady=5)

btn_french = tk.Button(lang_selection, text="Français", width=15, command=lambda: select_language('fr'))
btn_french.pack(pady=5)

# Fonction pour configurer l'interface graphique principale
def setup_gui():
    global root, btn_load, drop_label, lbl_video, progress_var, draw_boxes, show_plate_text, draw_cars, chk_draw_boxes, chk_show_plate_text, chk_draw_cars, gif_label, logo_label, github_logo_label

    root = TkinterDnD.Tk()  # Utiliser TkinterDnD pour le glisser-déposer
    root.title(texts['title'])
    root.geometry("800x600")  # Définir les dimensions initiales de la fenêtre

    # Ajouter une icône à la fenêtre
    icon_path = 'icons/logo.ico'
    root.iconbitmap(icon_path)

    # Créer un Frame pour contenir les logos
    logo_frame = tk.Frame(root)
    logo_frame.pack(side="top", anchor="nw", padx=10, pady=10)

    # Ajouter le logo de l'interface
    logo_path = 'icons/logo-interface.jpg'
    logo_img = Image.open(logo_path)
    logo_img = logo_img.resize((100, 100), Image.LANCZOS)  # Utiliser Image.LANCZOS pour une meilleure qualité
    logo_photo = ImageTk.PhotoImage(logo_img)
    
    logo_label = tk.Label(logo_frame, image=logo_photo)
    logo_label.image = logo_photo  # Nécessaire pour garder une référence de l'image
    logo_label.pack(side="left")

    # Ajouter le logo GitHub à côté du logo de l'interface
    github_logo_path = 'icons/github-logo.png'
    github_img = Image.open(github_logo_path)
    github_img = github_img.resize((100, 100), Image.LANCZOS)  # Redimensionner pour correspondre à la taille du logo interface
    github_photo = ImageTk.PhotoImage(github_img)
    
    github_logo_label = tk.Label(logo_frame, image=github_photo, cursor="hand2")
    github_logo_label.image = github_photo  # Nécessaire pour garder une référence de l'image
    github_logo_label.pack(side="left", padx=10)

    # Fonction pour ouvrir le projet
    def open_github():
        webbrowser.open("https://github.com/AxelTeam007/CarsDetect")  # Ouvrir le projet

    # Associer la fonction open_github au clic sur le logo GitHub
    github_logo_label.bind("<Button-1>", lambda e: open_github())

    # Onglets
    notebook = ttk.Notebook(root)
    notebook.pack(expand=1, fill="both")
    
    # Frame pour l'onglet de traitement vidéo
    frame_video = ttk.Frame(notebook)
    notebook.add(frame_video, text=texts['video'])

    # Frame pour l'onglet des paramètres
    frame_settings = ttk.Frame(notebook)
    notebook.add(frame_settings, text=texts['settings'])

    # Bouton pour charger la vidéo
    btn_load = tk.Button(frame_video, text=texts['load_video'], command=load_video)
    btn_load.pack()

    # Charger et afficher le GIF
    gif_path = 'icons/cloudupload.gif'
    gif = Image.open(gif_path)
    gif_frames = [ImageTk.PhotoImage(img) for img in ImageSequence.Iterator(gif)]
    gif_label = tk.Label(frame_video)
    gif_label.pack(pady=10)

    def animate_gif(ind):
        frame = gif_frames[ind]
        ind += 1
        if ind == len(gif_frames):
            ind = 0
        gif_label.configure(image=frame)
        root.after(100, animate_gif, ind)

    root.after(0, animate_gif, 0)

    # Label pour le glisser-déposer
    drop_label = tk.Label(frame_video, text=texts['drop_video'], width=40, height=10, bg="lightgray")
    drop_label.pack(pady=20)

    # Associer le glisser-déposer au label
    drop_label.drop_target_register(DND_FILES)
    drop_label.dnd_bind('<<Drop>>', drop)

    # Label pour afficher la vidéo
    lbl_video = tk.Label(frame_video)

    # Barre de progression
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(frame_video, variable=progress_var, maximum=100)
    progress_bar.pack(fill=tk.X, padx=10, pady=10)

    # Checkboxes pour les paramètres
    draw_boxes = tk.BooleanVar(value=True)
    show_plate_text = tk.BooleanVar(value=True)
    draw_cars = tk.BooleanVar(value=True)  # Ajout de l'option pour encadrer les voitures

    chk_draw_boxes = tk.Checkbutton(frame_settings, text=texts['encadrer_plaques'], variable=draw_boxes)
    chk_draw_boxes.pack(anchor="w", padx=10, pady=5)

    chk_show_plate_text = tk.Checkbutton(frame_settings, text=texts['afficher_numeros'], variable=show_plate_text)
    chk_show_plate_text.pack(anchor="w", padx=10, pady=5)

    chk_draw_cars = tk.Checkbutton(frame_settings, text=texts['encadrer_voitures'], variable=draw_cars)  # Texte traduit ici
    chk_draw_cars.pack(anchor="w", padx=10, pady=5)

    root.mainloop()

# Démarrer la sélection de langue
lang_selection.mainloop()