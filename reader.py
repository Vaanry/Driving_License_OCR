from itertools import zip_longest
import pandas as pd
from PIL import Image
from preprocessing import get_position, get_type, rotation
import torch
from torchvision import transforms

def read_img(dl_image, model, processor, trocr_model):
    """Detects labels on a Driver's License (DL) image and recognizes text within them.

    Parameters
    ----------
    dl_image : PIL.Image.Image
        The driver's license image.
    model : YOLO trained model
        The model for object detection.
    processor : TrOCRProcessor
        The processor for text recognition.
    trocr_model : VisionEncoderDecoderModel
        The model for text recognition.

    Returns
    -------
    dict
        A dictionary containing recognized text grouped by labels.
    """
    results = model.predict(source=dl_image, augment=True, verbose=False)
    labels = ['Address', 'Class', 'DOB', 'Exp date', 'First name', 'Issue date', 'Last name', 'License number', 'Sex']
    boxes = results[0].boxes
    data = {key: [] for key in labels}
    for bounding_box in boxes:
        x1, y1, x2, y2 = bounding_box.xyxy[0].tolist()
        crop_img = dl_image.crop((x1, y1, x2, y2)).convert("RGB")
        crop_img = crop_img.resize((crop_img.width * 4, crop_img.height * 4))
        pixel_values = processor(images=crop_img, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values, max_new_tokens=4000)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        data[labels[int(bounding_box.cls)]].append(generated_text)
    return data


def text_extraction(file, class_model, rotate_model=None, model, processor, trocr_model):
    """Extract text from a Driver's License (DL) ID and save it as pd.DataFrame.

    Parameters
    ----------
    file : str
        Path to the DL ID file.
    class_model : torch.nn.Module
        The ResNet model for ID determination.
    rotate_model : torch.nn.Module
        The ResNet model for position determination.
    model : YOLO trained model
        The YOLO model.
    processor : TrOCRProcessor
        The TrOCRProcessor.
    trocr_model : VisionEncoderDecoderModel
        The VisionEncoderDecoderModel.

    Returns
    -------
    pd.DataFrame or str
        If the type of ID is 'id', returns a DataFrame of recognized text by labels.
        If the type is not 'id', returns an empty DataFrame or None.
    """
    if get_type(class_model, file)=='id':
        rotated_image = rotation(rotate_model, file) if rotate_model else Image.open(file)
        data_read = read_img(rotated_image, model, processor, trocr_model)
        max_length = max(map(len, data_read.values()))
        data_read['Folder'] = [file.split(os.path.sep)[-2]] * max_length
        data = pd.DataFrame(list(zip_longest(*list(data_read.values()), fillvalue=None)), columns=list(data_read))
        return data if not data.empty else None
    else:
        return None