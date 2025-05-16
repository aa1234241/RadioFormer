from transformers import TimesformerConfig, TimesformerModel, TimesformerForVideoClassification
from transformers.models.timesformer.modeling_timesformer import TimesformerEmbeddings
from timm.models.registry import register_model
import torch
import torch.utils.checkpoint
from calflops import calculate_flops
@register_model
def timesformer(**kwargs):
    configuration = TimesformerConfig(image_size=112,
                                      patch_size=16,
                                      num_channels=8,
                                      num_frames=14,
                                      num_labels=7)
    model = TimesformerForVideoClassification.from_pretrained(
        "/home/xiaoyubai/.cache/huggingface/hub/models--facebook--timesformer-base-finetuned-k400/snapshots/8aaf40ea7d3d282dcb0a5dea01a198320d15d6c0")
    model.timesformer.embeddings = TimesformerEmbeddings(configuration)
    for i in range(model.timesformer.encoder.layer.__len__()):
        model.timesformer.encoder.layer[i].config = configuration

    model.classifier = torch.nn.Linear(in_features=768, out_features=7, bias=True)
    # model = TimesformerForVideoClassification(configuration)
    return model

if __name__ == '__main__':
    model = timesformer()
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=(1,14,8,112,112),
                                          output_as_string=True,
                                          output_precision=4)