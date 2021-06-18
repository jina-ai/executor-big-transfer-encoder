import os
import shutil

import PIL.Image as Image
import numpy as np

from jina import Flow, Document

cur_dir = os.path.dirname(os.path.abspath(__file__))


def data_generator(num_docs):
    for i in range(num_docs):
        doc = Document(
            uri=os.path.join(cur_dir, '..', 'data', 'test_image.png'))
        doc.convert_image_uri_to_blob()
        img = Image.fromarray(doc.blob.astype('uint8'))
        img = img.resize((96, 96))
        img = np.array(img).astype('float32') / 255
        doc.blob = img
        yield doc


def model_test(model_name, num_docs):
    shutil.rmtree('pretrained')
    os.environ['TRANSFER_MODEL_NAME'] = model_name
    with Flow.load_config('flow.yml') as flow:
        data = flow.post(on='/index', inputs=data_generator(5), 
                         request_size=num_docs)
        docs = data[0].docs
        for doc in docs:
            assert doc.embedding is not None


def test_all_models():
    for model_name in ['R50x1', 'R101x1', 'R50x3', 'R101x3', 'R152x4']:
        model_test(model_name, 10)
