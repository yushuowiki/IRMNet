import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models
from model.Pyconvs import PyConvBlock

# masked global pooling -- drop height and wedth
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask # mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area 
    return supp_feat # scala 
  
def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class IRMNet(nn.Module):
    def __init__(self, layers=50, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
        pretrained=True, sync_bn=True,shot=1, ppm_scales=[60, 30, 15, 8], vgg=True):
        super(IRMNet, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm        
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales # [60, 30, 15, 8]
        self.vgg = vgg
        self.convs=PyConvBlock(256,200,BatchNorm)
        D=10
        models.BatchNorm = BatchNorm
        
        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('INFO: Using ResNet {}'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512       

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        ) 
        # self.cls2 = nn.Sequential(
        #     nn.Conv2d(3, reduce_dim, kernel_size=3, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=0.1),                 
        #     nn.Conv2d(reduce_dim, classes, kernel_size=1)
        # )
        # self.change = nn.Sequential(
        #     nn.Conv2d(1, 2, kernel_size=1)
        # )                
        # 改变通道 到512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        ) 
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        ) 
        # [60, 30, 15, 8]
        self.pyramid_bins = ppm_scales # 
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )


        factor = 1
        mask_add_num = 11 # ablation 1 else 11
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []        
        for bin in self.pyramid_bins: #  # [60, 30, 15, 8] reduce_dim=256
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))                      
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))            
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))            
        self.init_merge = nn.ModuleList(self.init_merge) 
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)                             

        self.s_down = nn.Sequential(
            nn.Conv2d(256, D, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),  
            # nn.Dropout2d(p=0.5)                        
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )                        
     
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))     
        self.alpha_conv = nn.ModuleList(self.alpha_conv)
     


    def forward(self, x, s_x=torch.FloatTensor(1,1,3,473,473).cuda(), s_y=torch.FloatTensor(1,1,473,473).cuda(), y=None,pred=None):
        '''
        x: input query x
        s_x:support x
        s_y:support mask
        pred:
        ''' 
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)  # middle
            query_feat_3 = self.layer3(query_feat_2)  # high
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)
        query_feat = torch.cat([query_feat_3, query_feat_2], 1) # middle
        query_feat = self.down_query(query_feat) # 改变通道 到512
        

        #   Support Feature     
        supp_feat_list = []
        final_supp_list = []
        supp_mid_feat_list=[]
        mask_list = [] # for each shot
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1) # middle
                supp_feat_3 = self.layer3(supp_feat_2) # middle
                # 利用插值方法，对输入的张量数组进行上\下采样操作，换句话说就是科学合理地改变数组的尺寸大小，尽量保持数据完整。
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list.append(supp_feat_4) # shot number/high-level
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1) # millde-level
            supp_feat = self.down_supp(supp_feat)
            supp_mid_feat_list.append(supp_feat )# millde-level
            supp_feat = Weighted_GAP(supp_feat, mask) # masked global pooling batch*channel*1*1
            supp_feat_list.append(supp_feat) # millde-level after MGP prototype shape is (batch*channel*1*1)

        #类过滤---------------------------------------------
        cosine_eps = 1e-7
        mult_quer=self.convs(query_feat)
        resize=supp_mid_feat_list[0].size(2)
        tmp_mask0 = F.interpolate(mask_list[0], size=(resize, resize), mode='bilinear', align_corners=True)
        supp_feat_high = supp_mid_feat_list[0]*tmp_mask0 
        for i in range(1, len(final_supp_list)): # shot number
            tmp_mask = F.interpolate(mask_list[i], size=(resize, resize), mode='bilinear', align_corners=True)
            supp_feat_high = supp_mid_feat_list[i]* tmp_mask+supp_feat_high # 累计mask+加
        supp_feat_high = supp_feat_high/ len(supp_mid_feat_list)
        # print(supp_feat_high.shape)
        s_fil=self.s_down(supp_feat_high) # 一个conv2d support               
        q = mult_quer # need to convert to multiple class features
        bsize, ch_sz, sp_sz, _ = s_fil.size()[:]
        tmp_supp = s_fil # support               
        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
        tmp_supp_norm = tmp_supp/(torch.norm(tmp_supp, 2, 1, True)+cosine_eps) # support normalization
        corrI=[] # for each class
        for i in range(20): # for each class 
                tmp_query = q[:,i*10:(i+1)*10,:,:]
                tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1) 
                tmp_query_norm = tmp_query/(torch.norm(tmp_query, 2, 1, True)+cosine_eps) # querry normalization
                # cosine similarity
                corr=(tmp_query_norm.permute(0, 2, 1)).matmul(tmp_supp_norm)#b,hw,hw
                corr = corr.clamp(min=0) 
                corr=corr.mean(dim=1,keepdim=True) #b,1,hw
                corrI.append(corr)
        corrI=torch.cat(corrI,dim=1)
        corr_mean=corrI.mean(2) # reduce dimension
        corr_index=corr_mean.argmax(1)
        similarity=[]
        for j in range(bsize):
                index=corr_index[j]
                simi=q[j,index*10:(index+1)*10,:,:].unsqueeze(0)
                # simi=corrI[j].unsqueeze(0)[:,corr_index[j],:].unsqueeze(1)
                similarity.append(simi)
        similarity=torch.cat(similarity,dim=0)
        corr_query = similarity.view(bsize, 10, sp_sz, sp_sz) # class filter similarity
        # Relation Reference Module -----------------------------------------------

        rela_query_mask_list=[]
        for i, tmp_supp_feat in enumerate(final_supp_list): # short number, support high level feature
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask 
            #关系参考
            rela_query=query_feat_4
            rela_supp=tmp_supp_feat_4
            b,ch,rh,_=rela_query.size()[:]
            rela_query=rela_query.contiguous().view(b, ch, -1)
            rela_query_norm = torch.norm(rela_query, 2, 1, True)
            rela_supp = rela_supp.contiguous().view(b, ch, -1) 
            rela_supp = rela_supp.contiguous().permute(0, 2, 1) 
            rela_supp_norm = torch.norm(rela_supp, 2, 2, True)
            simi_rela=torch.bmm(rela_supp, rela_query)/(torch.bmm(rela_supp_norm, rela_query_norm) + cosine_eps)
            simi_rela = simi_rela.max(1)[0].view(b, rh*rh)   
            simi_rela = (simi_rela - simi_rela.min(1)[0].unsqueeze(1))/(simi_rela.max(1)[0].unsqueeze(1) - simi_rela.min(1)[0].unsqueeze(1) + cosine_eps)
            rela_query = simi_rela.view(b, 1, rh, rh)
            rela_query = F.interpolate(rela_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            rela_query_mask_list.append(rela_query)
        rela_query_mask = torch.cat(rela_query_mask_list, 1).mean(1).unsqueeze(1)     
        rela_query_mask = F.interpolate(rela_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True) 

        if self.shot > 1:
            supp_feat = supp_feat_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat += supp_feat_list[i]
            supp_feat /= len(supp_feat_list) # 平均值

        out_list = []
        pyramid_feat_list = [] # Multi-scale Interact Module output

        for idx, tmp_bin in enumerate(self.pyramid_bins): # [60, 30, 15, 8] interpolate size
            if tmp_bin <= 1.0: # ratio of the previous scale
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat) # [60, 30, 15, 8] size pool
            # supp_feat shape (batch, channel, 1, 1)
            supp_feat_bin = supp_feat.expand(-1, -1, bin, bin) # expand只能扩展维度为1的维度
            # -- corr_query
            corr_mask_bin = F.interpolate(corr_query, size=(bin, bin), mode='bilinear', align_corners=True) # -->(batch,channel,bin,bin)
            rela_mask_bin = F.interpolate(rela_query_mask, size=(bin, bin), mode='bilinear', align_corners=True) # -->(batch,channel,bin,bin)
            # -- corr_mask_bin
            # query_feat_bin 最初的query feature
            # rela_mask_bin 关系参考模块输出
            # supp_feat_bin millde-level after MGP and average over each shot
            # corr_mask_bin 类过滤输出
            merge_feat_bin = torch.cat([query_feat_bin,rela_mask_bin, supp_feat_bin, corr_mask_bin], 1) # (batch,final_channel,bin,bin)
            # 1x1 conv
            merge_feat_bin = self.init_merge[idx](merge_feat_bin) # for bin in [60, 30, 15, 8] combine and conv 

            if idx >= 1: # number th
                pre_feat_bin = pyramid_feat_list[idx-1].clone() # last element
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True) # uniform the height and wedth
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1) # combine this and the last one in channel level 
                # 1x1 conv
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin  

            # 3x3 conv +1 padding (size not change)
            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin  
            # 3x3 conv +1 padding (size not change)
            inner_out_bin = self.inner_cls[idx](merge_feat_bin) # size is bin
            
            out_list.append(inner_out_bin) # 多个输出
            # reshape to the origin size of input
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
            
            pyramid_feat_list.append(merge_feat_bin) # same size of input
            
        query_feat = torch.cat(pyramid_feat_list, 1) # connect all bin size
        query_feat = self.res1(query_feat) # conv2d 1x1 
        query_feat = self.res2(query_feat) + query_feat # conc2d 3x3 + padding 1           
        out = self.cls(query_feat) # 3x3 1x1
        
        #   Output Part
        if self.training:
            if self.zoom_factor != 1:
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            last_out=out
            main_loss = self.criterion(last_out, y.long())
            aux_loss = torch.zeros_like(main_loss).cuda()

            for idx_k in range(len(out_list)):
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y.long())   
            aux_loss = aux_loss / len(out_list)
            return last_out.max(1)[1], main_loss, aux_loss
        else: # not use interpolate
            zero=torch.zeros(size=(1,1,60,60)).cuda(non_blocking=True)
            gene_fg=torch.where(pred>0,pred,zero) # diffusion generation positive
            gene_bg=torch.where(pred<0,pred,zero) # diffusion generation negative
            gene=torch.cat([gene_fg, gene_bg], 1) 
            last_out=out + gene 
            last_out = F.interpolate(last_out, size=(h, w), mode='bilinear', align_corners=True)
            return last_out