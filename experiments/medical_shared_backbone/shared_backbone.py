"""FOCUS mechanisms on the benchmark's unmodified ResUNet32-mid modules."""
from types import MethodType
import torch
from torch.nn import functional as F
from base_network import ResUnet

class FocusResUNet(ResUnet):
    def __init__(self, source_sparse_class):
        super().__init__(1,[32,64,128,256])
        self.dropout_p=.9
        self.encoder_frozen=False
        self.sparsify_features=MethodType(source_sparse_class.sparsify_features,self)

    def encoder_modules(self):
        return (self.conv1,self.bn1,self.conv2,self.residual_conv_1,self.residual_conv_2,self.bridge)

    def first(self,x):
        return self.conv2(F.relu(self.bn1(self.conv1(x))))

    def decode(self,x1,x2,x3,x4):
        x6=self.up_residual_conv1(torch.cat([self.upsample_1(x4),x3],dim=1))
        x8=self.up_residual_conv2(torch.cat([self.upsample_2(x6),x2],dim=1))
        x10=self.up_residual_conv3(torch.cat([self.upsample_3(x8),x1],dim=1))
        return self.last(x10)

    def forward(self,x,layer_name_for_feature_extraction=None):
        x1=self.first(x)
        if self.training:x1=self.sparsify_features(x1,during_normal_training=True)
        x2=self.residual_conv_1(x1)
        if self.training:x2=self.sparsify_features(x2,during_normal_training=True)
        x3=self.residual_conv_2(x2)
        if self.training:x3=self.sparsify_features(x3,during_normal_training=True)
        x4=self.bridge(x3)
        out=self.decode(x1,x2,x3,x4)
        if layer_name_for_feature_extraction is not None:
            assert layer_name_for_feature_extraction=='encoder'
            return out,[self.sparsify_features(v) for v in (x1,x2,x3,x4)]
        return out

    def feature_forward(self,features_and_skips):
        assert self.encoder_frozen
        x1=features_and_skips[0];x2=self.residual_conv_1(x1);x3=self.residual_conv_2(x2);x4=self.bridge(x3)
        return self.decode(x1,x2,x3,x4)

    def freeze_layers(self,name):
        assert name=='encoder';self.encoder_frozen=True
        for module in self.encoder_modules():module.requires_grad_(False);module.eval()

    def train(self,mode=True):
        super().train(mode)
        if getattr(self,'encoder_frozen',False):
            for module in self.encoder_modules():module.eval()
        return self
