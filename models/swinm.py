import torch.nn as nn
import timm
from functools import partial
from timm.models.layers import  create_classifier
from timm.models.registry import register_model
import torch.nn.functional as F
__all__ = ['Swinm']  # model_registry will add each entrypoint fn to this
from calflops import calculate_flops


class Swinm(nn.Module):
    def __init__(self, pretrained=False, img_size=112):
        super(Swinm, self).__init__()
        self.vf = timm.create_model('swin_small_patch4_window7_224', pretrained=True, img_size=224)
        self.global_pool, self.fc = create_classifier(self.vf.num_features * 8, 7, pool_type='avg')
        self.phase_fc = nn.Linear(self.vf.num_features * 8, 7)

    def forward(self, x):
        x = x.reshape(-1, 3, x.shape[3], x.shape[4])
        x = self.vf.forward_features(x)
        x = x.permute(0, 2, 1)
        fc_f = x.reshape(-1, x.shape[1] * 8, x.shape[2])
        fc_i = F.adaptive_avg_pool1d(fc_f, 1)[:, :, 0]
        out = self.phase_fc(fc_i)
        return out,out

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)


@register_model
def swinm_s(pretrained=False, **kwargs):
    return Swinm(pretrained=pretrained)

if __name__ == '__main__':


    model  = swinm_s()
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=(1, 96, 3, 224, 224),
                                          output_as_string=True,
                                          output_precision=4)


    print('test')