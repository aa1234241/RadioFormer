from transformers import VivitConfig, VivitModel, VivitForVideoClassification
from transformers.models.vivit.modeling_vivit import VivitEmbeddings
from timm.models.registry import register_model
import torch
import torch.utils.checkpoint
from torch import nn
from calflops import calculate_flops


@register_model
def vivit(**kwargs):
    # Initializing a ViViT google/vivit-b-16x2-kinetics400 style configuration
    configuration = VivitConfig(image_size=112,
                                num_frames=14,
                                tubelet_size=[2, 16, 16],
                                num_channels=8,
                                num_labels=7)
    # # Initializing a model (with random weights) from the google/vivit-b-16x2-kinetics400 style configuration
    # model = VivitForVideoClassification(configuration)
    model = VivitForVideoClassification.from_pretrained(
        "/home/xiaoyubai/.cache/huggingface/hub/models--google--vivit-b-16x2-kinetics400/snapshots/8a7171a57f79b9aaa58bc8d977c002a0ea0f0d42")
    model.vivit.embeddings = VivitEmbeddings(configuration)
    model.classifier = torch.nn.Linear(in_features=768, out_features=7, bias=True)
    # model.cuda()
    return model


if __name__ == '__main__':
    model = vivit()
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=(1,14,8,112,112),
                                          output_as_string=True,
                                          output_precision=4)
