from ultralytics import YOLO 

class YOLO_network():
    
    #Definizione del modello: finora è stato usato il modello MEDIUM
    model_path = r'dresscode\network_output\training_v3.2_small_shuffled\weights\best.pt'
    conf_file = r'datasets\hands_inspection_v3\data_shuffled.yaml'
    output_dir = r'dresscode\network_output'
    name_output_dir = r'training_v3.2.1_small_shuffled_rs14'
    patience=30
    imgsz=640
    epochs=200

    def __init__(self):
        self.model = YOLO(self.model_path)

    def train_network(self, conf_file, patience, imgsz, epochs, output_dir, name_output_dir):
        self.model.train(
            data=conf_file,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            patience=patience,
            device=0,
            amp=False, #Necessario in quanto se è attivo si blocca la GPU
            save=True,
            plots=True,
            project=output_dir,
            name=name_output_dir,
            verbose=True,
            show=True
        )

    def tracker_offline(self, source, show=False):
        self.model.track(source=source, device='cpu', show=show, save=True, tracker='bytetrack.yaml')

    def print_metrics(self, conf_file):
        metrics = self.model.val(data=conf_file, split='test', device='cpu')
        print("\n--- METRICHE PRINCIPALI ---")
        for k, v in metrics.results_dict.items():
            print(f"{k}: {v:.4f}")
        
        print("\n--- mAP PER CLASSE ---")
        for cls_idx, mAP in enumerate(metrics.maps):
            print(f"{metrics.names[cls_idx]}: {mAP:.4f}")

    def predict_images(self, src_folder, output_dir, name_output_dir):
        """
        Aggiungere descrizione - metodo naive per valutare il modello
        """
        images = self.model.predict(
            source=src_folder,
            conf=0.25,
            device='cpu',
            show=True,
            save=True,
            project=output_dir,
            name=name_output_dir
            )

    def main(self):
        src = r'dresscode\testing_files\testing_video\IMG_4615.MOV'
        self.tracker_offline(src, True)
        
if __name__ == '__main__':
    network = YOLO_network()
    network.main()