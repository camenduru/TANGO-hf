import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx

# ----------- 1 full conv-based encoder------------- #
"""
from tm2t
TM2T: Stochastical and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts
https://github.com/EricGuo5513/TM2T
"""
from .quantizer import *
from .layer import *

class SCFormer(nn.Module):
    def __init__(self, args):
        super(VQEncoderV3, self).__init__()


        n_down = args.vae_layer
        channels = [args.vae_length]
        for i in range(n_down-1):
            channels.append(args.vae_length)
        
        input_size = args.vae_test_dim
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)
    def forward(self, inputs): # bs t n
        '''
        face 51 or 106
        hand 30*(15)
        upper body 
        lower body 
        global 1*3 
        max length around 180 --> 450
        '''
        bs, t, n = inputs.shape
        inputs = inputs.reshape(bs*t, n)
        inputs = self.spatial_transformer_encoder(inputs) # bs*t c
        cs = inputs.shape[1]
        inputs = inputs.reshape(bs, t, cs).permute(0, 2, 1).reshape(bs*cs, t)
        inputs = self.temporal_cnn_encoder(inputs) # bs*c t
        ct = inputs.shape[1]
        outputs = inputs.reshape(bs, cs, ct).permute(0, 2, 1) # bs ct cs
        return outputs

class VQEncoderV3(nn.Module):
    def __init__(self, args):
        super(VQEncoderV3, self).__init__()
        n_down = args.vae_layer
        channels = [args.vae_length]
        for i in range(n_down-1):
            channels.append(args.vae_length)
        
        input_size = args.vae_test_dim
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs

class VQEncoderV6(nn.Module):
    def __init__(self, args):
        super(VQEncoderV6, self).__init__()
        n_down = args.vae_layer
        channels = [args.vae_length]
        for i in range(n_down-1):
            channels.append(args.vae_length)
        
        input_size = args.vae_test_dim
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs

class VQEncoderV4(nn.Module):
    def __init__(self, args):
        super(VQEncoderV4, self).__init__()
        n_down = args.vae_layer
        channels = [args.vae_length]
        for i in range(n_down-1):
            channels.append(args.vae_length)
        
        input_size = args.vae_test_dim
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs

class VQEncoderV5(nn.Module):
    def __init__(self, args):
        super(VQEncoderV5, self).__init__()
        n_down = args.vae_layer
        channels = [args.vae_length]
        for i in range(n_down-1):
            channels.append(args.vae_length)
        
        input_size = args.vae_test_dim
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs

class VQDecoderV4(nn.Module):
    def __init__(self, args):
        super(VQDecoderV4, self).__init__()
        n_up = args.vae_layer
        channels = []
        for i in range(n_up-1):
            channels.append(args.vae_length)
        channels.append(args.vae_length)
        channels.append(args.vae_test_dim)
        input_size = args.vae_length
        n_resblk = 2
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            up_factor = 2 if i < n_up - 1 else 1
            layers += [
                nn.Upsample(scale_factor=up_factor, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs

class VQDecoderV5(nn.Module):
    def __init__(self, args):
        super(VQDecoderV5, self).__init__()
        n_up = args.vae_layer
        channels = []
        for i in range(n_up-1):
            channels.append(args.vae_length)
        channels.append(args.vae_length)
        channels.append(args.vae_test_dim)
        input_size = args.vae_length
        n_resblk = 2
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            up_factor = 2 if i < n_up - 1 else 1
            layers += [
                #nn.Upsample(scale_factor=up_factor, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs

class VQDecoderV7(nn.Module):
    def __init__(self, args):
        super(VQDecoderV7, self).__init__()
        n_up = args.vae_layer
        channels = []
        for i in range(n_up-1):
            channels.append(args.vae_length)
        channels.append(args.vae_length)
        channels.append(args.vae_test_dim+4)
        input_size = args.vae_length
        n_resblk = 2
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            up_factor = 2 if i < n_up - 1 else 1
            layers += [
                #nn.Upsample(scale_factor=up_factor, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs
    
class VQDecoderV3(nn.Module):
    def __init__(self, args):
        super(VQDecoderV3, self).__init__()
        n_up = args.vae_layer
        channels = []
        for i in range(n_up-1):
            channels.append(args.vae_length)
        channels.append(args.vae_length)
        channels.append(args.vae_test_dim)
        input_size = args.vae_length
        n_resblk = 2
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs

class VQDecoderV6(nn.Module):
    def __init__(self, args):
        super(VQDecoderV6, self).__init__()
        n_up = args.vae_layer
        channels = []
        for i in range(n_up-1):
            channels.append(args.vae_length)
        channels.append(args.vae_length)
        channels.append(args.vae_test_dim)
        input_size = args.vae_length * 2
        n_resblk = 2
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                # nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs


# -----------2 conv+mlp based fix-length input ae ------------- #
from .layer import reparameterize, ConvNormRelu, BasicBlock
"""
from Trimodal,
encoder:
    bs, n, c_in --conv--> bs, n/k, c_out_0 --mlp--> bs, c_out_1, only support fixed length
decoder:
    bs, c_out_1 --mlp--> bs, n/k*c_out_0 --> bs, n/k, c_out_0 --deconv--> bs, n, c_in
"""
class PoseEncoderConv(nn.Module):
    def __init__(self, length, dim, feature_length=32):
        super().__init__()
        self.base = feature_length
        self.net = nn.Sequential(
            ConvNormRelu(dim, self.base, batchnorm=True), #32
            ConvNormRelu(self.base, self.base*2, batchnorm=True), #30
            ConvNormRelu(self.base*2, self.base*2, True, batchnorm=True), #14     
            nn.Conv1d(self.base*2, self.base, 3)
        )
        self.out_net = nn.Sequential(
            nn.Linear(12*self.base, self.base*4),  # for 34 frames
            nn.BatchNorm1d(self.base*4),
            nn.LeakyReLU(True),
            nn.Linear(self.base*4, self.base*2),
            nn.BatchNorm1d(self.base*2),
            nn.LeakyReLU(True),
            nn.Linear(self.base*2, self.base),
        )
        self.fc_mu = nn.Linear(self.base, self.base)
        self.fc_logvar = nn.Linear(self.base, self.base)

    def forward(self, poses, variational_encoding=None):
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        out = self.net(poses)
        out = out.flatten(1)
        out = self.out_net(out)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar


class PoseDecoderFC(nn.Module):
    def __init__(self, gen_length, pose_dim, use_pre_poses=False):
        super().__init__()
        self.gen_length = gen_length
        self.pose_dim = pose_dim
        self.use_pre_poses = use_pre_poses

        in_size = 32
        if use_pre_poses:
            self.pre_pose_net = nn.Sequential(
                nn.Linear(pose_dim * 4, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
            in_size += 32

        self.net = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, gen_length * pose_dim),
        )

    def forward(self, latent_code, pre_poses=None):
        if self.use_pre_poses:
            pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
            feat = torch.cat((pre_pose_feat, latent_code), dim=1)
        else:
            feat = latent_code
        output = self.net(feat)
        output = output.view(-1, self.gen_length, self.pose_dim)
        return output


class PoseDecoderConv(nn.Module):
    def __init__(self, length, dim, use_pre_poses=False, feature_length=32):
        super().__init__()
        self.use_pre_poses = use_pre_poses
        self.feat_size = feature_length
        
        if use_pre_poses:
            self.pre_pose_net = nn.Sequential(
                nn.Linear(dim * 4, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
            self.feat_size += 32

        if length == 64:
            self.pre_net = nn.Sequential(
                nn.Linear(self.feat_size, self.feat_size),
                nn.BatchNorm1d(self.feat_size),
                nn.LeakyReLU(True),
                nn.Linear(self.feat_size, self.feat_size//8*64),
            )
        elif length == 34:
            self.pre_net = nn.Sequential(
                nn.Linear(self.feat_size, self.feat_size*2),
                nn.BatchNorm1d(self.feat_size*2),
                nn.LeakyReLU(True),
                nn.Linear(self.feat_size*2, self.feat_size//8*34),
            )
        elif length == 32:
            self.pre_net = nn.Sequential(
                nn.Linear(self.feat_size, self.feat_size*2),
                nn.BatchNorm1d(self.feat_size*2),
                nn.LeakyReLU(True),
                nn.Linear(self.feat_size*2, self.feat_size//8*32),
            )
        else:
            assert False
        self.decoder_size = self.feat_size//8
        self.net = nn.Sequential(
            nn.ConvTranspose1d(self.decoder_size, self.feat_size, 3),
            nn.BatchNorm1d(self.feat_size),
            nn.LeakyReLU(0.2, True),
            
            nn.ConvTranspose1d(self.feat_size, self.feat_size, 3),
            nn.BatchNorm1d(self.feat_size),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(self.feat_size, self.feat_size*2, 3),
            nn.Conv1d(self.feat_size*2, dim, 3),
        )

    def forward(self, feat, pre_poses=None):
        if self.use_pre_poses:
            pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
            feat = torch.cat((pre_pose_feat, feat), dim=1)
        #print(feat.shape)
        out = self.pre_net(feat)
        #print(out.shape)
        out = out.view(feat.shape[0], self.decoder_size, -1)
        #print(out.shape)
        out = self.net(out)
        out = out.transpose(1, 2)
        return out

'''
Our CaMN Modification
'''
class PoseEncoderConvResNet(nn.Module):
    def __init__(self, length, dim, feature_length=32):
        super().__init__()
        self.base = feature_length
        self.conv1=BasicBlock(dim, self.base, reduce_first = 1, downsample = False, first_dilation=1) #34
        self.conv2=BasicBlock(self.base, self.base*2, downsample = False, first_dilation=1,) #34
        self.conv3=BasicBlock(self.base*2, self.base*2, first_dilation=1, downsample = True, stride=2)#17            
        self.conv4=BasicBlock(self.base*2, self.base, first_dilation=1, downsample = False)
        
        self.out_net = nn.Sequential(
            # nn.Linear(864, 256),  # for 64 frames
            nn.Linear(17*self.base, self.base*4),  # for 34 frames
            nn.BatchNorm1d(self.base*4),
            nn.LeakyReLU(True),
            nn.Linear(self.base*4, self.base*2),
            nn.BatchNorm1d(self.base*2),
            nn.LeakyReLU(True),
            nn.Linear(self.base*2, self.base),
        )

        self.fc_mu = nn.Linear(self.base, self.base)
        self.fc_logvar = nn.Linear(self.base, self.base)

    def forward(self, poses, variational_encoding=None):
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        out1 = self.conv1(poses)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out = self.conv4(out3)
        out = out.flatten(1)
        out = self.out_net(out)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar    
    

# -----------3 lstm ------------- #
'''
bs, n, c_int --> bs, n, c_out or bs, 1 (hidden), c_out 
'''
class AELSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.motion_emb = nn.Linear(args.vae_test_dim, args.vae_length)
        self.lstm = nn.LSTM(args.vae_length, hidden_size=args.vae_length, num_layers=4, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.out = nn.Sequential(
            nn.Linear(args.vae_length, args.vae_length//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(args.vae_length//2, args.vae_test_dim)
        )
        self.hidden_size = args.vae_length

    def forward(self, inputs):
        poses = self.motion_emb(inputs)  
        out, _ = self.lstm(poses)
        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        out_poses = self.out(out) 
        return {
            "poses_feat":out,
            "rec_pose": out_poses,
            }     
    
class PoseDecoderLSTM(nn.Module):
    """
    input bs*n*64
    """
    def __init__(self,pose_dim, feature_length):
        super().__init__()
        self.pose_dim = pose_dim
        self.base = feature_length
        self.hidden_size = 256
        self.lstm_d = nn.LSTM(self.base, hidden_size=self.hidden_size, num_layers=4, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.out_d = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size // 2, self.pose_dim)
        )

    def forward(self, latent_code):
        output, _ = self.lstm_d(latent_code)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs
        #print("outd:", output.shape)
        output = self.out_d(output.reshape(-1, output.shape[2]))
        output = output.view(latent_code.shape[0], latent_code.shape[1], -1)
        #print("resotuput:", output.shape)
        return output
     
# ---------------4 transformer --------------- #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print(self.pe.shape, x.shape)
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)  

class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.skelEmbedding = nn.Linear(args.vae_test_dim, args.vae_length)
        self.sequence_pos_encoder = PositionalEncoding(args.vae_length, 0.3)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=args.vae_length,
                                                          nhead=4,
                                                          dim_feedforward=1025,
                                                          dropout=0.3,
                                                          activation="gelu",
                                                          batch_first=True
                                                         )
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=4)
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, inputs):
        x = self.skelEmbedding(inputs)  #bs * n * 128
        #print(x.shape)
        xseq = self.sequence_pos_encoder(x)
        device = xseq.device
        #mask = self._generate_square_subsequent_mask(xseq.size(1)).to(device)
        final = self.seqTransEncoder(xseq)
        #print(final.shape)
        mu = final[:, 0:1, :]
        logvar = final[:, 1:2, :]
        return final, mu, logvar
    
class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vae_test_len = args.vae_test_len
        self.vae_length = args.vae_length
        self.sequence_pos_encoder = PositionalEncoding(args.vae_length, 0.3)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=args.vae_length,
                                                          nhead=4,
                                                          dim_feedforward=1024,
                                                          dropout=0.3,
                                                          activation="gelu",
                                                          batch_first=True)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=4)
        self.finallayer = nn.Linear(args.vae_length, args.vae_test_dim)
        
    def forward(self, inputs):
        timequeries = torch.zeros(inputs.shape[0], self.vae_test_len, self.vae_length, device=inputs.device) 
        timequeries = self.sequence_pos_encoder(timequeries)
        output = self.seqTransDecoder(tgt=timequeries, memory=inputs)
        output = self.finallayer(output)
        return output    