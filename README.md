# Moire_python_project

This repository supports the research of multi-exposure images effect on Moire detection.
- hdr: contains a script to transform images under different exposures into a HDR image.
- detection_model: contains a script used to perform transfer learning of VGG-19 model to classify Moire and non-Moire images.
- grad_cam: contains a script to extract Gradient Cam from VGG-19 model given an input image to classify Moire class. Additionally, with gradio_demo notebook containing the deployment of Gradient Cam visualization, we can observe the feature extract on Moire image which helps the model to classify input.

There will be an additional requirements.txt file under each folders.
