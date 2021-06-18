import shutil

import pytest
import os
import numpy as np
import PIL.Image as Image

from jina import DocumentArray, Document

from jinahub.image.encoder.big_transfer import BigTransferEncoder

directory = os.path.dirname(os.path.realpath(__file__))


def test_initialization_and_model_download():
    shutil.rmtree('pretrained')
    # This call will download the model
    encoder = BigTransferEncoder()
    assert encoder.channel_axis == 1
    assert encoder.model_path == 'pretrained'
    assert encoder.model_name == 'R50x1'
    assert not encoder.on_gpu
    assert os.path.exists('pretrained')
    assert os.path.exists(os.path.join('pretrained', 'saved_model.pb'))
    # This call will use the downloaded model
    _ = BigTransferEncoder()


def test_encoding():
    doc = Document(uri=os.path.join(directory, '../data/test_image.png'))
    doc.convert_image_uri_to_blob()
    img = Image.fromarray(doc.blob.astype('uint8'))
    img = img.resize((96, 96))
    img = np.array(img).astype('float32') / 255
    doc.blob = img
    assert doc.embedding is None

    encoder = BigTransferEncoder()

    encoded_doc = encoder.encode(DocumentArray([doc]))
    assert encoded_doc[0].embedding is not None