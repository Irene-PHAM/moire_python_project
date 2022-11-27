

# All imports
import urllib
from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as F
from torchcam.methods import GradCAMpp
#from grad_cam_function import GradCAMpp
from typing import List, Tuple, Dict
import numpy as np
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(path_to_model, device):
    if device.type == 'cpu':
        model = torch.load(path_to_model, map_location=torch.device('cpu'))
    else:
        model = torch.load(path_to_model)
    return model.eval()

transform_test = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        ])
    
    

class GradCam:
    """
    GradCam class to perform grad-cam algorithm on input images to visualise extracted features used
    used for classification task.
    """
    def __init__(self, model, img_path_list) -> None:
        """ Initialise with loaded VGG-19 model and list of images
        Parameters:
            model: loaded VGG-19 model
            img_path_list: list of Moire images under various exposure
        """
        self.model = model
        self.img_path_list = img_path_list
        # Use GradCAMpp model from torchcam
        # 'features' layer is the default layers of VGG-19 model that extracting features
        self.cam_extractor = GradCAMpp(self.model, 'features')
    
    @staticmethod
    def img_to_tensor(img_path:str) -> Tensor:
        """ Transform image to tensor
        Parameters:
            img_path : path of the image
        Return:
            Image tensor resize into 224*224 size tensor
        """
        img = Image.open(urllib.request.urlopen(img_path))
        input_tensor =  transform_test(img).to(device)
        return input_tensor

    def get_layered_cam_array(self) -> Tuple:
        """ Get a CAM array from the combination of highlighted part of CAM from single exposure image
        Return:
            A final layered CAM with pixels of the highest value from the input images
        """
        # Get list of CAM array
        list_of_cams = self.get_multi_image_cam_array_list()
        # Get the flatten list
        list_of_cam_flattened = [list(np.concatenate(cam).flat) for cam in list_of_cams]
        # Get the highest pixel value from all layers in each respective position and combine them
        # together for the final layer
        layered_tuple=list(map(max, zip(*[tuple(cam) for cam in list_of_cam_flattened]))) 
        return np.reshape(layered_tuple, (-1, 7))
    def count_highlighted_pixel(self) -> Dict:
        """ Count the highlighted pixel (with value >0.5) from each input image and the fused layer
        Return:
            Dictionary of name of the image as key and count of highlighted pixel as value
        """
        multi_cam_array_list = self.get_multi_image_cam_array_list()
        multi_count = {k:(self.calculate_bright_pixel(v)) for k,v in zip(self.img_path_list, multi_cam_array_list )}
        layered_cam_array = self.get_layered_cam_array()
        layered_cam_array_count = self.calculate_bright_pixel(layered_cam_array)
        multi_count['fused_layer'] =  layered_cam_array_count 
        return multi_count

    @staticmethod
    def calculate_bright_pixel(array_list: List) -> int:
        """ Count number of highlighted pixel in the array, highlighted pixel is considered as 
        value that is >= 0.5
        Return:
            Count of the pixel >=0.5
        """
        count = 0
        for array in array_list:
            count +=len(list(filter (lambda x : x >= 0.5, array)))
        return count    
    
    def get_multi_image_cam_array_list(self) -> List:
        """ Get a list of CAM in array from input images
        Return: List of CAM arrays
        """
        tensor_list = self.get_multi_image_cam_tensor_list()
        return [tensor.squeeze(0).detach().cpu().numpy() for tensor in tensor_list]
        
    def get_multi_image_cam_tensor_list(self) -> List:
        """ Get a list of CAM in tensor from input images
        Return: List of CAM tensors
        """
        tensor_list = []
        for img in self.img_path_list:
            input_tensor = self.img_to_tensor(img)
            out = self.model(input_tensor.unsqueeze(0))
            cams = self.cam_extractor(0,out)
            tensor_list.append(cams[0])
        return tensor_list

