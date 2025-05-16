from monai.networks.nets.swin_unetr import SwinTransformer
import torch
from timm.models.registry import register_model
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['Swin3dmodel']
from calflops import calculate_flops

class Swin3dmodel(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, window_size=(7, 7, 7), patch_size=(2, 2, 2), depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24),**kwargs):
        super(Swin3dmodel, self).__init__()

        self.swin3d = SwinTransformer(in_chans=in_chans, embed_dim=embed_dim, window_size=window_size,
                                      patch_size=patch_size, depths=depths, num_heads=num_heads)

        self.fc = nn.Linear(384*8,7)

    def forward(self, x):
        x = x.reshape(-1,1,x.shape[-3],x.shape[-2],x.shape[-1])
        # x = self.swin3d(x)[4]

        x = self.swin3d(x)

        # x = F.adaptive_avg_pool3d(x,1)[:,:,0,0,0]
        # x = x.reshape(-1, x.shape[1]*8)
        # x = self.fc(x)
        return x

@register_model
def swin3dmodel(**kwargs):
    return Swin3dmodel(**kwargs)


if __name__ == '__main__':
    print('test')
    device = 'cuda'
    # model = SwinTransformer(in_chans=1, embed_dim=24, window_size=(7, 7, 7), patch_size=(2, 2, 2), depths=(2, 2, 2, 2),
    #                         num_heads=(3, 6, 12, 24))

    # model  = swin3dmodel(depths=(2, 2, 6, 2))
    model = SwinTransformer(in_chans=1, embed_dim=96, window_size=(7, 7, 7), patch_size=(2, 2, 2), depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24))

    a = torch.rand(1,1,128,128,128)
    model = model.cuda()

    sb = model(a.cuda())



    # shape = (1, 8, 32, 128, 128)
    # sample_input = torch.rand(shape, device=device)
    # out = model(sample_input)[4]
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=(1, 1, 224, 224, 224),
                                          output_as_string=True,
                                          output_precision=4)


    print('test')
