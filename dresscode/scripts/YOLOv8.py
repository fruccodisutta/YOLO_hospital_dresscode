from ultralytics import YOLO 

class YOLO_network():
    
    #Definizione del modello: finora è stato usato il modello MEDIUM
    model = YOLO(r'ultralytics\training_output\training_v2\weights\best.pt')
    mode = None

    def train_network(self, conf_file, device, patience, imgsz, epochs, output_dir, name_output_dir):
        self.model.train(
            data=conf_file,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            patience=patience,
            device=device,
            amp=False, #Necessario in quanto se è attivo si blocca la GPU
            save=True,
            project=output_dir,
            name=name_output_dir,
            verbose=True,
            show=True
        )

    def print_metrics(self, conf_file):
        metrics = self.model.val(data=conf_file, split='test', device='cpu')
        print("\n--- METRICHE PRINCIPALI ---")
        for k, v in metrics.results_dict.items():
            print(f"{k}: {v:.4f}")
        
        print("\n--- mAP PER CLASSE ---")
        for cls_idx, mAP in enumerate(metrics.maps):
            print(f"{metrics.names[cls_idx]}: {mAP:.4f}")

    def main(self):
        conf_file = r'datasets\hands_inspection_v3\data.yaml'
        output_dir = r'ultralytics\training_output'
        name_output_dir = r'training_v3'
        patience=20
        imgsz=640
        epochs=200
        device='cpu' #inserire 'cpu' per usare la CPU, 0 per usare la GPU

        self.print_metrics(conf_file=conf_file)
        
if __name__ == '__main__':
    network = YOLO_network()
    network.main()