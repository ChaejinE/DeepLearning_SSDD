from tf_2.official.saved_model import TF2SavedModel
from utils.file_utils import convert_image_bytes

import numpy as np


class ClassifierTF:
    def __init__(self, config):
        self.config = config
        self.pb_dir = self.parse_config("pb_dir")
        self._model = self._build_model()

    def _build_model(self):
        try:

            return TF2SavedModel(pb_dir=self.pb_dir)
        except TypeError as te:
            print('ClassifierTF Model is not built.')
            print(te)

            raise
        except OSError as oe:
            print('ClassifierTF Model is not built.')
            print(oe)

            raise

    def parse_config(self, key):
        if key in self.config:

            return self.config[key]
        return None

    def inference(self, image):
        image = np.array(image)
        image = convert_image_bytes(image)
        predict_id = self._model.inference(image)

        return predict_id

    def is_defect(self, image):
        predict_id = self.inference(image)
        if predict_id == 1:

            return False
        else:

            return True

    def classify(self, image, boxes):
        image = np.array(image)
        boxes = np.array(boxes, dtype=np.int)
        defects = []
        for box in boxes:
            if box[0] == box[2] or box[1] == box[3]:
                
                continue
            _image = image[box[1]:box[3], box[0]:box[2], :]
            if self.is_defect(_image):
                defects.append(box.tolist())

        return defects
