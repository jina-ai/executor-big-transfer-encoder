__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import numpy as np
import tensorflow as tf
from typing import Optional

from jina import DocumentArray, Executor, requests
from jina.excepts import PretrainedModelFileDoesNotExist
from jina.logging.predefined import default_logger


class BigTransferEncoder(Executor):
    """
    :class:`BigTransferEncoder` is Big Transfer (BiT) presented by
    Google (https://github.com/google-research/big_transfer).
    Uses pretrained BiT to encode data from a ndarray, potentially
    B x (Channel x Height x Width) into a ndarray of `B x D`.
    Internally, :class:`BigTransferEncoder` wraps the models from
    https://storage.googleapis.com/bit_models/.

    .. warning::

        Known issue: this does not work on tensorflow==2.2.0,
        https://github.com/tensorflow/tensorflow/issues/38571

    :param model_path: the path of the model in the `SavedModel` format.
        The pretrained model can be downloaded at
        wget https://storage.googleapis.com/bit_models/Imagenet21k/[model_name]/feature_vectors/saved_model.pb
        wget https://storage.googleapis.com/bit_models/Imagenet21k/[model_name]/feature_vectors/variables/variables.data-00000-of-00001
        wget https://storage.googleapis.com/bit_models/Imagenet21k/[model_name]/feature_vectors/variables/variables.index

        ``[model_name]`` includes `R50x1`, `R101x1`, `R50x3`, `R101x3`, `R152x4`

        The `model_path` should be a directory path, which has the following structure.

        .. highlight:: bash
         .. code-block:: bash

            .
            ├── saved_model.pb
            └── variables
                ├── variables.data-00000-of-00001
                └── variables.index

        :param channel_axis: the axis id of the channel, -1 indicate the color
            channel info at the last axis. If given other, then `
            `np.moveaxis(data, channel_axis, -1)`` is performed before :meth:`encode`.
    """
    def __init__(self,
                 model_path: Optional[str] = 'pretrained',
                 channel_axis: int = 1,
                 on_gpu: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_axis = channel_axis
        self.model_path = model_path
        self.on_gpu = on_gpu
        self.logger = default_logger

        if self.model_path and os.path.exists(self.model_path):
            cpus = tf.config.experimental.list_physical_devices(
                device_type='CPU')
            gpus = tf.config.experimental.list_physical_devices(
                device_type='GPU')
            if self.on_gpu and len(gpus) > 0:
                cpus.append(gpus[0])
            tf.config.experimental.set_visible_devices(devices=cpus)
            self.logger.info(f'BiT model path: {self.model_path}')
            from tensorflow.python.keras.models import load_model
            _model = load_model(self.model_path)
            self.model = _model.signatures['serving_default']
            self._get_input = tf.convert_to_tensor
        else:
            raise PretrainedModelFileDoesNotExist(
                f'model at {self.model_path} does not exist')

    @requests
    def encode(self, docs: DocumentArray, **kwargs) -> DocumentArray:
        """
        Encode data into a ndarray of `B x D`.
        Where `B` is the batch size and `D` is the Dimension.

        :param docs: an array in size `B`
        :return: an ndarray in size `B x D`.
        """
        data = np.zeros((docs.__len__(),) + docs[0].blob.shape)
        for index, doc in enumerate(docs):
            data[index] = doc.blob
        if self.channel_axis != -1:
            data = np.moveaxis(data, self.channel_axis, -1)
        _output = self.model(self._get_input(data.astype(np.float32)))
        output = _output['output_1'].numpy()
        for index, doc in enumerate(docs):
            doc.embedding = output[index]
        return docs