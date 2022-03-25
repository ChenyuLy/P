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


# class Patch_Embed(nn.Module):
#     def __init__(self, patch_size=16, image_size=224, cin_dim=3, embed_dim=768):
#         super(Patch_Embed, self).__init__()
#         self.img_size = (image_size, image_size)
#         self.patch_size = (patch_size, patch_size)
#         self.grid_size = (image_size // patch_size, image_size // patch_size)
#         self.proj = nn.Conv2d(cin_dim, embed_dim, self.patch_size, self.patch_size)
#         self.patches_num = self.grid_size[0] * self.grid_size[1]
#
#     def forward(self, x):
#         B, C, W, H = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         # flatten: [B, C, H, W] -> [B, C, HW]
#         # transpose: [B, C, HW] -> [B, HW, C] [B,patches_num ,embed_dim]
#         x = self.proj(x).flatten(2).transpose(1, 2)  # 经过一次卷积以后 输入图片的维度变为[batchsize ,patches_num , embed_dim]
#
#         return x
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)  # 把分下来的小框口整合到了patch维度
    return windows


def Token_reduction(x, window_size, token_num):#(8,14*14,16*16*3)
    B, P, D = x.shape
    x = x.reshape(B,token_num,token_num,window_size,window_size,3).permute(0,5,1,3,2,4)
    x = x.reshape(B,3,token_num*window_size,token_num*window_size)

    return x


class Patch_Embed(nn.Module):
    def __init__(self, patch_size=16, image_size=224, cin_dim=3, embed_dim=768):
        super(Patch_Embed, self).__init__()
        self.img_size = (image_size, image_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.proj = nn.Conv2d(cin_dim, embed_dim, self.patch_size, self.patch_size)
        self.patches_num = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        B, C, W, H = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C] [B,patches_num ,embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)  # 经过一次卷积以后 输入图片的维度变为[batchsize ,patches_num , embed_dim]

        return x


class Additon_conv2d(nn.Module):
    def __init__(self, dim, window_size, token_num, hidden_rate=2,patch_embed = Patch_Embed):
        super(Additon_conv2d, self).__init__()
        self.window_size = window_size
        self.token_num = token_num
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3)
        # self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=7, padding=3)
        self.patch_embed = patch_embed( patch_size=16, image_size=224, cin_dim=3, embed_dim=768)

    def forward(self, x):

        cls,x_ = x.split((1,196),dim = 1)
        x_ = Token_reduction(x_, window_size=self.window_size,
                            token_num=self.token_num)
        x_ = self.conv1(x_)
        # x_ = self.conv2(x_)
        x_ = self.patch_embed(x_)
        x = torch.cat((cls, x_), dim=1)
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
        self.attn = Attention(dim=dim, num_heads =self.num_head )
        self.mlp = MLP(in_feature=dim, hidden_feature=4 * mlp_hidden_ratio)
        self.norm2 = norm_layer(dim)
        self.addition_block = Additon_conv2d(3, 16, 14 )

    def forward(self, x):

        x = x + self.addition_block(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Vison_transformer(nn.Module):
    def __init__(self, dim=768, depth=12, img_size=224, patch_size=16, in_c=3, num_classes=1000, embed_lay=Patch_Embed,
                 num_heads=12,
                 ):
        super(Vison_transformer, self).__init__()
        self.dim = dim
        self.class_num = num_classes
        self.depth = depth
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_c = in_c
        self.Blocks = nn.Sequential(*[Block(dim=dim) for i in range(depth)])
        self.Embed_lay = embed_lay(patch_size=patch_size, embed_dim=dim)
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
        x = x + self.pos_embed
        x = self.Blocks(x)

        x = self.norm(x)
        x = self.head(x)  # [B,num_patches+1,class]
        return x[:, 0]


def vit_base_patch16_224(num_classes: int = 1000):
    model = Vison_transformer(num_classes=num_classes)
    return model


# def vit_base_patch32_224(num_classes: int = 1000): #Mar18_14-12-20_PT6630W
#     model = Vison_transformer(num_classes=num_classes,patch_size=32,dim=3072)
#     return model

def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = Vison_transformer(img_size=224,
                              patch_size=32,
                              dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model
