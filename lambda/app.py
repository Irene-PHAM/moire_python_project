import json
import torch
from grad_cam_url import GradCam, load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def serverless_pipeline(device, model_path='model.pt'):
    """Initializes the model and tokenzier and returns a predict function that ca be used as pipeline"""
    model = load_model(model_path, device)
    def predict(image_path_list):
        """predicts the answer on an given question and context. Uses encode and decode method from above"""
        grad_cam = GradCam(model, image_path_list)
        result = grad_cam.count_highlighted_pixel()
        return result
    return predict

# initializes the pipeline
grad_cam_pipeline = serverless_pipeline(device)

def handler(event, context):
    try:
        # loads the incoming event into a dictonary
        print('Received event: ' + json.dumps(event, indent=2))
        img_list = event['urls']
        # uses the pipeline to predict the answer
        result = grad_cam_pipeline(image_path_list=img_list)
        return json.dumps({'result': result})
        
    except Exception as e:
        print(repr(e))
        return  json.dumps({"error": repr(e)})
