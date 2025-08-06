import os
import pillow_heif
from PIL import Image

def create_empty_txt(images_dir, labels_dir, extensions=None):
    """
    Per ogni immagine in images_dir crea un file .txt vuoto in labels_dir con lo stesso nome (ma estensione .txt).
    Se labels_dir non esiste, viene creato.
    Questo step è necessario in quanto la YOLOv8 intepreta le immagini senza bounding box come negative 
    """

    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    for filename in os.listdir(images_dir):
        nome_base, ext = os.path.splitext(filename)
        if ext.lower() in extensions:
            txt_path = os.path.join(labels_dir, nome_base + '.txt')
            if not os.path.exists(txt_path):
                with open(txt_path, 'w') as f:
                    pass  # crea file vuoto
                print(f"Creato file vuoto: {txt_path}")
            else:
                print(f"File già esistente: {txt_path}")

def heic_to_jpeg(input_folder, output_folder):
    """
    Aggiungere descrizione
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)    

    for file in os.listdir(input_folder):
        if file.lower().endswith(".heic"):
            heif_file = pillow_heif.read_heif(os.path.join(input_folder, file))
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data
            )
            output_path = os.path.join(output_folder, file.replace(".heic", ".jpg"))
            image.save(output_path, format="JPEG")
            print(f"Convertita: {file} -> {output_path}")
