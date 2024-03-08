import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init

patch_size_1 = 13
patch_size_2 = 7

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MConv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_0=3,  stride=1,  ):
        super(MConv1, self).__init__()
        # Groupwise Convolution kernel_size=3
        self.gwc1 = nn.Conv2d(in_channels, 32, kernel_size=7, padding=7 // 2, stride=stride, groups=8)
        # Groupwise Convolution kernel_size=5
        self.gwc2 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=5 // 2, stride=stride, groups=8)
        # self.gwc3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, padding=5 // 2, stride=stride, groups=8)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, 16, kernel_size=1, stride=stride, groups=8)

    def forward(self, x):
        a = self.gwc1(x)
        b = self.gwc2(x)
        c = self.pwc(x)
        x = torch.cat((a, b, c), dim=1)
        # x = torch.cat((x, c), dim=1)

        return x

class MConv2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MConv2, self).__init__()
        # Groupwise Convolution kernel_size=3
        self.gwc1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2, stride=stride, groups=8)
        # Groupwise Convolution kernel_size=5
        self.gwc2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, padding=5 // 2, stride=stride, groups=8)
        # self.gwc3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, padding=5 // 2, stride=stride, groups=8)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=8)

    def forward(self, x):
        a = self.gwc1(x)
        b = self.gwc2(x)
        # d = self.gwc3(x)
        c = self.pwc(x)
        return a + b + c

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)
        self.Conv1d = nn.Sequential(
            nn.Conv1d(9, 5, kernel_size=3, padding=1),
            nn.BatchNorm1d(5),
            nn.ReLU(),
        )
        self.to_q_1 = nn.Sequential(
                    nn.Conv2d(heads, heads, kernel_size=3, padding=1),
                    nn.BatchNorm2d(heads),
                    nn.ReLU(),
                )
        self.to_k_1 = nn.Sequential(
            nn.Conv2d(heads, heads, kernel_size=5, padding=2),
            nn.BatchNorm2d(heads),
            nn.ReLU(),
        )
        self.to_v_1 = nn.Sequential(
            nn.Conv2d(heads, heads, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(heads),
            nn.ReLU(),
        )
        self.to_q_2 = nn.Sequential(
            nn.Conv2d(heads, heads, kernel_size=3, padding=1),
            nn.BatchNorm2d(heads),
            nn.ReLU(),
        )
        self.to_k_2 = nn.Sequential(
            nn.Conv2d(heads, heads, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(heads),
            nn.ReLU(),
        )
        self.to_v_2 = nn.Sequential(
            nn.Conv2d(heads, heads, kernel_size=5, padding=2),
            nn.BatchNorm2d(heads),
            nn.ReLU(),
        )
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x):
        x1 = x[:, 0:9, :]
        x2 = x[:, 9:, :]
        x1 = self.Conv1d(x1)
        x1 = x1.reshape(x1.shape[0], self.heads, x1.shape[1], -1)
        x2 = x2.reshape(x2.shape[0], self.heads, x2.shape[1], -1)
        # ---------------------------!1!-----------------------------
        q1, k1, v1 = self.to_q_1(x1), self.to_k_1(x1), self.to_v_1(x1)
        dots1 = torch.einsum('bhid,bhjd->bhij', q1, k1) * self.scale
        attn1 = dots1.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        # ---------------------------!2!-----------------------------
        q2, k2, v2 = self.to_q_2(x2), self.to_k_2(x2), self.to_v_2(x2)
        dots2 = torch.einsum('bhid,bhjd->bhij', q2, k2) * self.scale
        attn2 = dots2.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out1 = torch.einsum('bhij,bhjd->bhid', attn1, v2)  # product of v times whatever inside softmax
        out1 = rearrange(out1, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out1 = self.nn1(out1)
        out1 = self.do1(out1)
        out2 = torch.einsum('bhij,bhjd->bhid', attn2, v1)  # product of v times whatever inside softmax
        out2 = rearrange(out2, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out2 = self.nn1(out2)
        out2 = self.do1(out2)

        out = out1 + out2
        return out

#!!!!!!!!!!!!!!!!!!
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout)),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x


class TNCCA(nn.Module):
    def __init__(self, NC, num_classes, in_channels=1, num_tokens_1=8, num_tokens_2=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(TNCCA, self).__init__()
        self.L1 = num_tokens_1
        self.L2 = num_tokens_2
        self.cT = dim
        self.m1_conv2d_features = nn.Sequential(
            MConv1(NC*8, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(), )
        self.m2_conv2d_features = nn.Sequential(
            MConv2(NC*4, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(), )
        self.conv3d_features_1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv3d_features_2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=4, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        self.conv2d_features_1 = nn.Sequential(
            nn.Conv2d(NC*8, out_channels=64, kernel_size=(3, 3), groups=2, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2d_features_2 = nn.Sequential(
            nn.Conv2d(NC*4, out_channels=64, kernel_size=(3, 3), groups=2, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2d_cc = nn.Sequential(
            nn.Conv2d(64+8*(NC-2), out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2d_kk = nn.Sequential(
            nn.Conv2d(64, out_channels=64, kernel_size=(3, 3), groups=8, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Tokenization
        self.token_wA_1 = nn.Parameter(torch.empty(1, self.L1, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA_1)
        self.token_wA_2 = nn.Parameter(torch.empty(1, self.L2, 64),
                                       requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA_2)

        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding_1 = nn.Parameter(torch.empty(1, (num_tokens_1 + 1), dim))
        torch.nn.init.normal_(self.pos_embedding_1, std=.02)
        self.pos_embedding_2 = nn.Parameter(torch.empty(1, (num_tokens_2 + 1), dim))
        torch.nn.init.normal_(self.pos_embedding_2, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)


    def forward(self, x1, x2):

        x1 = self.conv3d_features_1(x1)
        x2 = self.conv3d_features_2(x2)

        x1 = x1.reshape(x1.shape[0], -1, patch_size_1, patch_size_1)
        x2 = x2.reshape(x2.shape[0], -1, patch_size_2, patch_size_2)
        x1 = self.m1_conv2d_features(x1)
        x2 = self.m2_conv2d_features(x2)

        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = rearrange(x2, 'b c h w -> b (h w) c')

        wa_1 = rearrange(self.token_wA_1, 'b h w -> b w h')  # Transpose
        A_1 = torch.einsum('bij,bjk->bik', x1, wa_1)
        A_1 = rearrange(A_1, 'b h w -> b w h')  # Transpose
        A_1 = A_1.softmax(dim=-1)
        VV_1 = torch.einsum('bij,bjk->bik', x1, self.token_wV)
        T_1 = torch.einsum('bij,bjk->bik', A_1, VV_1)
        cls_tokens_1 = self.cls_token.expand(x1.shape[0], -1, -1)
        x_1 = torch.cat((cls_tokens_1, T_1), dim=1)
        x_1 += self.pos_embedding_1
        x_1 = self.dropout(x_1)

        wa_2 = rearrange(self.token_wA_2, 'b h w -> b w h')  # Transpose
        A_2 = torch.einsum('bij,bjk->bik', x2, wa_2)
        A_2 = rearrange(A_2, 'b h w -> b w h')  # Transpose
        A_2 = A_2.softmax(dim=-1)
        VV_2 = torch.einsum('bij,bjk->bik', x2, self.token_wV)
        T_2 = torch.einsum('bij,bjk->bik', A_2, VV_2)
        cls_tokens_2 = self.cls_token.expand(x2.shape[0], -1, -1)
        x_2 = torch.cat((cls_tokens_2, T_2), dim=1)
        x_2 += self.pos_embedding_2
        x_2 = self.dropout(x_2)

        x = torch.cat((x_1, x_2), dim=1)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x


if __name__ == '__main__':
    model = TNCCA(30, 16)
    model.eval()
    print(model)
    input_1 = torch.randn(32, 1, 30, 13, 13)
    input_2 = torch.randn(32, 1, 30, 7, 7)
    y = model(input_1, input_2)
    print(y.size())


