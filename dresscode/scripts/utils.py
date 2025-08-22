import os
import pillow_heif
import random 
import shutil
from pathlib import Path
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


def shuffle(dataset_dir, output_dir, train_ratio, val_ratio, test_ratio):
    """
    Aggiungere descrizione
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    train_ratio = train_ratio
    val_ratio = val_ratio
    test_ratio = test_ratio 
    max_name_len = 100  # lunghezza massima consentita per i nomi file (senza estensione)
    random.seed(14)

    # Funzione per compatibilità long path su Windows
    def winpath(p: Path):
        return f"\\\\?\\{os.path.abspath(p)}"

    # Carica tutte le coppie (immagine, label) in una lista
    pairs = []
    for split in ["train", "valid", "test"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        for img_path in img_dir.glob("*.*"):
            label_path = lbl_dir / (img_path.stem + ".txt")
            if label_path.exists():
                pairs.append((img_path, label_path))
            else:
                print(f"[ATTENZIONE] Label mancante per {img_path}")

    print(f"Totale file trovati: {len(pairs)}")

    # Shuffle
    random.shuffle(pairs)

    # Calcolo gli indici di split
    total = len(pairs)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * val_ratio)

    train_pairs = pairs[:train_end]
    valid_pairs = pairs[train_end:valid_end]
    test_pairs = pairs[valid_end:]

    # Funzione per salvare i file nelle nuove cartelle
    def save_split(pairs, split_name):
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        for img_path, label_path in pairs:
            # Rinomina se il nome file è troppo lungo
            new_name = img_path.name
            if len(img_path.stem) > max_name_len:
                new_name = img_path.stem[:max_name_len] + img_path.suffix
            
            # Copia immagine
            shutil.copy2(winpath(img_path), winpath(img_out / new_name))
            # Copia label
            shutil.copy2(winpath(label_path), winpath(lbl_out / (Path(new_name).stem + ".txt")))

    # Salvataggio finale
    save_split(train_pairs, "train")
    save_split(valid_pairs, "valid")
    save_split(test_pairs, "test")

    print(f"Nuovo dataset mescolato creato in: {output_dir}")


if __name__ == "__main__":
    src_dir = r"datasets\hands_inspection_v3"
    output_dir = r"datasets\hands_inspection_v3\shuffled_rs14"
    shuffle(src_dir, output_dir, 0.8, 0.1, 0.1)