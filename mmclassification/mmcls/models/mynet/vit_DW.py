import torch.nn as nn
import torch
from functools import partial


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class Patch_Embed(nn.Module):
    def __init__(self, patch_size=16, image_size=224, cin_dim=3, embed_dim=768, dw=2, norm_layer=None):
        super(Patch_Embed, self).__init__()
        self.dw = dw
        if self.dw is not None:

            self.depthwise = nn.Conv2d(cin_dim, cin_dim, kernel_size=7, padding=3)
            self.depthwise = nn.Conv2d(cin_dim, cin_dim, kernel_size=7, padding=3, groups=cin_dim)
            self.pointwise = nn.Conv2d(cin_dim, cin_dim, kernel_size=1)
            self.img_size = image_size
        else:
            self.img_size = image_size
        self.img_size = (self.img_size, self.img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.proj = nn.Conv2d(cin_dim, embed_dim, self.patch_size, self.patch_size)
        self.patches_num = self.grid_size[0] * self.grid_size[1]
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self, x):
        if self.dw is not None:
            x = x + self.pointwise(self.depthwise(x))
        # if self.dw is not None:
        #     x = x + self.depthwise(x)
        B, C, W, H = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C] [B,patches_num ,embed_dim]

        x = self.proj(x).flatten(2).transpose(1, 2)  # 经过一次卷积以后 输入图片的维度变为[batchsize ,patches_num , embed_dim]
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # token的维数
                 num_heads=12,
                 ):
        super(Attention, self).__init__()
        self.heads_num = num_heads
        self.head_dim = dim // num_heads
        self.qk_scale = self.head_dim ** -0.5  # 后续做softmax时候用到
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, P, D = x.shape  # [batch,patch,dim] #每个头运算中 做运算的维度为 patch 和dim

        qkv = self.qkv(x).reshape(B, P, 3, D // self.heads_num, self.heads_num).permute(2, 0, 4, 1, 3)
        # qkv [batch,patch,3dim] -> [batch,patch,3,dim/head,head]->[3,batch,head,patch,dim-head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Q[patch,dim]@K[dim,patch]
        atten = q @ k.transpose(-2, -1) * self.qk_scale
        atten = atten.softmax(dim=-1)
        x = (atten @ v).transpose(-2, -3).reshape(B, P,
                                                  D)  # atten @ v    [batch,patch,head,dim-head][batch,patch,dim]
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_feature, out_feature=None, hidden_feature=None, nonLinear_func=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        self.drop = drop
        self.in_feature = in_feature
        self.out_feature = in_feature or out_feature
        self.hidden_feature = hidden_feature
        self.fc1 = nn.Linear(self.in_feature, self.hidden_feature)
        self.activation = nonLinear_func()
        self.fc2 = nn.Linear(self.hidden_feature, self.in_feature)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_head=12,
                 mlp_hidden_ratio=4,
                 drop=0.,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.num_head = num_head
        self.att_dim = dim
        self.attn = Attention(dim=dim)
        self.mlp = MLP(in_feature=dim, hidden_feature=4 * mlp_hidden_ratio)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Vison_transformer(nn.Module):
    def __init__(self, dim=768, depth=12, img_size=224, patch_size=16, in_c=3, num_classes=1000, embed_lay=Patch_Embed,
                 dw=2
                 ):
        super(Vison_transformer, self).__init__()
        self.dim = dim
        self.class_num = num_classes
        self.depth = depth
        self.img_size = img_size / dw
        self.patch_size = patch_size
        self.in_c = in_c
        self.Blocks = nn.Sequential(*[Block(dim=dim) for i in range(depth)])
        self.Embed_lay = embed_lay(dw=dw)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.Embed_lay.patches_num + 1, dim))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(self.dim)

        # 分类头
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        x = self.Embed_lay(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.Blocks(x)

        x = self.norm(x)
        x = self.head(x)  # [B,num_patches+1,class]
        return x[:, 0]


def vit_base_DW_patch16_224(num_classes: int = 1000):
    model = Vison_transformer(num_classes=num_classes)
    return model
