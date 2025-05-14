import av
import torch
import numpy as np
from transformers import VideoMAEConfig, VideoMAEForVideoClassification,VideoMAEModel
from transformers.models.videomae.modeling_videomae import VideoMAEEmbeddings
from timm.models.registry import register_model
import torch
import torch.utils.checkpoint
from calflops import calculate_flops

from transformers import VideoMAEForVideoClassification
@register_model
def videomaed(**kwargs):
    configuration = VideoMAEConfig(image_size=112,
                                      patch_size=16,
                                      num_channels=8,
                                      num_frames=14,
                                      num_labels=7)
    model = VideoMAEForVideoClassification.from_pretrained(
        "/home/xiaoyubai/.cache/huggingface/hub/models--MCG-NJU--videomae-base-finetuned-kinetics/snapshots/488eb9a0565f257b32866000305c8178965eb9f6")
    model.videomae.embeddings = VideoMAEEmbeddings(configuration)
    model.classifier = torch.nn.Linear(in_features=768, out_features=7, bias=True)
    return model

if __name__ == '__main__':
    model = videomaed()
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=(1,14,8,112,112),
                                          output_as_string=True,
                                          output_precision=4)
