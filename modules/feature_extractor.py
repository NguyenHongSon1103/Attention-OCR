"""
Input shape: (batch_size, w, h, c) , eg (64, 224, 224, 3)
Output shape: (batch_size, length), eg(64, 512)
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import applications as apps, models, layers

class ResNet50:
    """
    Some output layer: 

    conv5_block1_add

    conv5_block2_add

    conv5_block3_add

    Input: RGB image
    Output: a tensor with shape ( None, WxH, 2048)

    """
    def __init__(self, cutted_layer_name='conv5_block3_add', input_shape=(None, None, 3)):
        base_model = apps.ResNet50(include_top=False, input_shape=input_shape)
        if cutted_layer_name is not None:
            output = base_model.get_layer(cutted_layer_name).output
        else:
            output = base_model.output       
        self.channel = output.shape[-1]
        output = layers.Reshape((-1, self.channel))(output)
        self.model = models.Model(inputs=base_model.input, outputs=output)
    def __call__(self, tensor_img):
        return self.model(tensor_img)

class EfficientNetB0:
    """
    block7a_se_squeeze (GlobalAvera (None, 1152)
    top_activation (Activation)     (None, None, None, 1280)
    Input: RGB image
    Output: a tensor with shape ( None, WxH, 1280)

    """
    def __init__(self, cutted_layer_name='top_activation', input_shape=(None, None, 3)):
        base_model = apps.EfficientNetB0(include_top=False, input_shape=input_shape)
        if cutted_layer_name is not None:
            output = base_model.get_layer(cutted_layer_name).output
        else:
            output = base_model.output
        self.channel = output.shape[-1]
        output = layers.Reshape((-1, self.channel))(output)
        self.model = models.Model(inputs=base_model.input, outputs=output)
    def __call__(self, tensor_img):
        return self.model(tensor_img)


if __name__ == '__main__':
    # Use for test shape
    import numpy as np

    batch_size = 10
    img_w = 416
    img_h = 64
    tensor = np.random.random((batch_size, img_w, img_h, 3))
    feature_extraction = EfficientNetB0(cutted_layer_name='top_activation', input_shape=(None, None, 3))
    out = feature_extraction(tensor)
    print(out)
    """
    From (10, 416, 64, 3) -> (10, 26, 1280)
    <==> (batch_size, img_w, img_h, 3) -> (batch_size, seq_len, depth)
    """