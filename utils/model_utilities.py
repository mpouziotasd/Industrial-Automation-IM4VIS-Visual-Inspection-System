from ultralytics import YOLO


def load_model(model_path=None):
    """
        Description:
        Returns the loaded Computer Vision model using Ultralytics
        
        Returns:
            model: YOLO
            status: int
    """
    if not model_path:
        print("Warning: Model path not set")
        return None, -2
    
    try:
        model = YOLO(model_path)
        # model.set_classes(['Storage Box'])
        return model, 0
    except Exception as e:
        print("Error when loading the model", e)
        return None, -1

def detect(model, frame, device='cpu'):
    """
        Inference using CPU or GPU
        Returns detections as an Ultraytics object using the loaded model.
    """
    results = model(frame, device=device)
    return results
