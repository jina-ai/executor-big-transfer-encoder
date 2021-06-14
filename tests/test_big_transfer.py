import pytest
import os
import numpy as np
import PIL.Image as Image

from jina import DocumentArray, Document
from jina.excepts import PretrainedModelFileDoesNotExist


from jinahub.image.encoder.big_transfer import BigTransferEncoder

directory = os.path.dirname(os.path.realpath(__file__))


def test_initialization():
    encoder = BigTransferEncoder()
    assert encoder.channel_axis == 1
    assert encoder.model_path == "pretrained"
    assert not encoder.on_gpu
    with pytest.raises(PretrainedModelFileDoesNotExist):
        encoder = BigTransferEncoder(model_path="wrong_path")


def test_encoding():
    doc = Document(uri=os.path.join(directory, 'test_image.png'))
    doc.convert_image_uri_to_blob()
    img = Image.fromarray(doc.blob.astype('uint8'))
    img = img.resize((96, 96))
    img = np.array(img).astype('float32') / 255
    doc.blob = img

    encoder = BigTransferEncoder()

    encoded_doc = encoder.encode(DocumentArray([doc]))

    assert encoded_doc[0].embedding