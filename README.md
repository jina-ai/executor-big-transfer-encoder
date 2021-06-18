<p align="center">
<img src="https://github.com/jina-ai/jina/blob/master/.github/logo-only.gif?raw=true" alt="Jina banner" width="200px">
</p>

# Big Transfer Image Encoder

### Description
The Big Transfer Encoder (BiT) uses the Big Transfer models presented by Google [here]((https://github.com/google-research/big_transfer)).
It uses a pretrained version of a BiT model to encode an image from an array of shape 
(Batch x (Channel x Height x Width)) into an array of shape (Batch x Encoding)

### Parameters
The following parameters can be used:

- `model_path` (string, default: "pretrained"): The folder where the downloaded pretrained Big Transfer model is located
- `model_name` (string, default: "R50x1"): The model to be downloaded when the model_path is empty. Choose from ['R50x1', 'R101x1', 'R50x3', 'R101x3', 'R152x4']
- `channel_axis` (int): The axis where the channel of the images needs to be (model-dependent)
- `on_gpu` (bool): Specifies whether the model should be used on GPU or CPU. To use GPU,
either the GPU docker container needs to be used or you need to install CUDA 11.3 and cudnn8 (similar versions might also work) 
  