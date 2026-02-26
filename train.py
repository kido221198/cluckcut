from ultralytics import YOLO


if __name__ == '__main__':
    # Create a new YOLO model from scratch
    model = YOLO("yolov8n-seg.yaml")

    # Load a pretrained YOLO model (recommended for training)
    # model = YOLO("pretrained_model/best_synth_data_0611_15-185.pt")
    # model = YOLO("runs/segment/train64/weights/best.pt")

    # Train the model using the YAML file
    # model.resume = True
    results = model.train(data="./cb.yaml", epochs=600)

    # Evaluate the model's performance on the validation set
    results = model.val(save_json=True)

    # Export the model to ONNX format
    success = model.export()