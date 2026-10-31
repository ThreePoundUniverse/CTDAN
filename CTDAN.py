import torch
import torch.nn as nn
from net_modules import *


class EEGNetEEG(nn.Module):
    def __init__(self, drop_out=0.5, kernLength=20, chans=60):
        super().__init__()
        F2 = chans * 2
        downSample_1, downSample_2 = 4, 5

        self.depth = [1, 2]
        self.stage0 = nn.Sequential(
            nn.Conv1d(chans, F2, kernLength, groups=chans),
        )
        self.stage1 = nn.ModuleList([])
        for _ in range(self.depth[0]):
            self.stage1.append(
                nn.Sequential(
                    nn.Conv1d(F2, F2, kernLength // 2, groups=F2),
                    nn.BatchNorm1d(F2),
                    nn.ReLU(),
                    nn.Dropout(drop_out)
                )
            )
        self.stage2 = nn.ModuleList([])
        for _ in range(self.depth[1]):
            self.stage2.append(
                nn.Sequential(
                    SeparableConv1d(F2, F2, kernel_size=kernLength, padding=kernLength // 2),
                    SeparableConv1d(F2, F2, kernel_size=kernLength, padding=kernLength // 2),
                    nn.BatchNorm1d(F2),
                    nn.ReLU(),
                    nn.Dropout(drop_out)
                )
            )
        self.merge_s1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
        )
        self.merge_s2 = nn.Sequential(
            nn.AvgPool1d(downSample_2),
            SKAttention1D(F2, reduction=4),
            Permute([0, 2, 1])
        )

    def forward(self, x):
        x = self.stage0(x)
        for stage1 in self.stage1:
            x = stage1(x)
        x = self.merge_s1(x)
        for stage2 in self.stage2:
            x = stage2(x)
        x = self.merge_s2(x)
        return x  # (B, 49, 120)


class NIRSNET_spatial(nn.Module):
    def __init__(self, drop_out=0.5, F1=30, D=2, F2=120, C=40, time_point=40):
        super().__init__()
        self.drop_out = drop_out
        self.block_1 = nn.Sequential(
            # left, right, up, bottom
            nn.ZeroPad2d((2, 1, 0, 0)),
            nn.Conv2d(1, F1, (1, 4), bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(),
        )
        self.block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 0, 0, 0)),
            nn.Conv2d(F1, D * F1, (1, 2), bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ReLU(),
            nn.Dropout()
        )
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 0, 0, 0)),
            nn.Conv2d(D * F1, F2, (1, 2), groups=D * F1, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((C, 1)),
            nn.Dropout(self.drop_out)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.squeeze(3)
        return x  # (B, 120, 40)


class Domain_adaption_module_model(nn.Module):
    def __init__(self, num_of_class, kernLength=20):
        super().__init__()
        self.eeg_extrator_s = EEGNetEEG(kernLength=kernLength)
        self.nirs_extrator_s = NIRSNET_spatial()

        self.classifier = nn.Sequential(
            nn.Linear(120 * 2, num_of_class),
        )

        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, 120))
        self.positon_eeg = nn.Parameter(torch.randn(1, 49 + 1, 120))

        self.cls_token_nirs = nn.Parameter(torch.randn(1, 1, 120))
        self.positon_nirs = nn.Parameter(torch.randn(1, 40 + 1, 120))

        self.transformer_eeg = TransformerVit(dim=120, depth=2, heads=8, dim_head=256, mlp_dim=128, dropout=0.)
        self.transformer_nirs = TransformerVit(dim=120, depth=2, heads=8, dim_head=256, mlp_dim=128, dropout=0.)

    def forward(self, s_eeg, s_nirs):
        eeg_feature_s = self.eeg_extrator_s(s_eeg)

        eeg_cls_tokens = self.cls_token_eeg.expand(eeg_feature_s.shape[0], -1, -1)
        atten_x_eeg = torch.cat((eeg_cls_tokens, eeg_feature_s), dim=1)
        atten_x_eeg += self.positon_eeg
        tr_output_eeg = self.transformer_eeg(atten_x_eeg)
        cls_token_eeg = tr_output_eeg[:, 0, :]  # (B, 120)

        nirs_feature_s = self.nirs_extrator_s(s_nirs).permute(0, 2, 1)
        nirs_cls_tokens = self.cls_token_nirs.expand(nirs_feature_s.shape[0], -1, -1)
        atten_x_nirs = torch.cat((nirs_cls_tokens, nirs_feature_s), dim=1)
        atten_x_nirs += self.positon_nirs
        tr_output_nirs = self.transformer_nirs(atten_x_nirs)
        cls_token_nirs = tr_output_nirs[:, 0, :]  # (B, 120)

        final_ = self.classifier(torch.cat((cls_token_eeg, cls_token_nirs), dim=1))
        return final_, cls_token_eeg, cls_token_nirs

    def forward_infer(self, s_eeg, s_nirs):
        # EEG branch
        eeg_feature_s = self.eeg_extrator_s(s_eeg)
        eeg_cls_tokens = self.cls_token_eeg.expand(eeg_feature_s.shape[0], -1, -1)
        atten_x_eeg = torch.cat((eeg_cls_tokens, eeg_feature_s), dim=1)
        atten_x_eeg += self.positon_eeg
        tr_output_eeg = self.transformer_eeg(atten_x_eeg)
        cls_token_eeg = tr_output_eeg[:, 0, :]

        # fNIRS branch
        nirs_feature_s = self.nirs_extrator_s(s_nirs).permute(0, 2, 1)
        nirs_cls_tokens = self.cls_token_nirs.expand(nirs_feature_s.shape[0], -1, -1)
        atten_x_nirs = torch.cat((nirs_cls_tokens, nirs_feature_s), dim=1)
        atten_x_nirs += self.positon_nirs
        tr_output_nirs = self.transformer_nirs(atten_x_nirs)
        cls_token_nirs = tr_output_nirs[:, 0, :]

        final_ = self.classifier(torch.cat((cls_token_eeg, cls_token_nirs), dim=1))
        return final_


# proposed
class MoCoV3(nn.Module):
    def __init__(self, num_of_class=3, m=0.999):
        super().__init__()
        self.m = m

        self.model = Domain_adaption_module_model(num_of_class)
        self.encoder_q = self.model
        self.encoder_k = Domain_adaption_module_model(num_of_class)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, source_eeg, target_eeg, source_nirs, target_nirs):
        out, feature_source_eeg, feature_source_nirs = self.model(source_eeg, source_nirs)
        _, feature_target_eeg, feature_target_nirs = self.model(target_eeg, target_nirs)
        _, feature_eeg, feature_nirs = self.model(
            torch.cat((source_eeg, target_eeg), dim=0),
            torch.cat((source_nirs, target_nirs), dim=0))

        q1 = nn.functional.normalize(feature_source_eeg, dim=1)
        q2 = nn.functional.normalize(feature_source_nirs, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k1, k2 = self.encoder_k(source_eeg, source_nirs)[1], self.encoder_k(source_eeg, source_nirs)[2]
        k1 = nn.functional.normalize(k1, dim=1)
        k2 = nn.functional.normalize(k2, dim=1)

        feature = torch.cat((feature_eeg, feature_nirs), dim=1)  # (2B, 240)
        feature_source = torch.cat((feature_source_eeg, feature_source_nirs), dim=1)  # (B, 240)
        feature_target = torch.cat((feature_target_eeg, feature_target_nirs), dim=1)  # (B, 240)

        return out, feature, q1, q2, k1, k2, [feature_source, feature_target]


class AdversarialNetwork(nn.Module):
    # domain discriminator: 240 -> 120 -> 120 -> 1
    def __init__(self, in_feature=240, hidden_size=120):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.act1 = nn.LeakyReLU(0.2)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.grl(x)
        x = self.layer1(x)
        x = self.bn(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        return self.sigmoid(x)


if __name__ == '__main__':
    # shape test
    eeg = torch.randn((16, 60, 1000))
    nirs = torch.randn((16, 40, 40))
    model = MoCoV3(num_of_class=3)
    out, feature, q1, q2, k1, k2, [a, b] = model(eeg, eeg, nirs, nirs)
    print(out.shape, feature.shape, q1.shape, q2.shape, k1.shape, k2.shape, a.shape, b.shape)
    # domain discriminator test
    ad_net = AdversarialNetwork(240, 120)
    d_out = ad_net(feature)
    print('discriminator output:', d_out.shape)
