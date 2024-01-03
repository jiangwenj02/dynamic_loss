import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_metanet import BaseMetaNet
from ..builder import METANETS

act_dict={'relu': nn.ReLU,
          'elu': nn.ELU}

@METANETS.register_module()
class DynamicLoss(BaseMetaNet):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 h_dim,
                 use_LC=True,
                 use_MG=False,
                 rank_num=0,
                 LC_classwise=False,
                 actfun='relu',
                 ):
        super(DynamicLoss, self).__init__()
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.rank_num = rank_num

        self.use_LC = use_LC
        self.use_MG = use_MG
        self.LC_classwise = LC_classwise

        self.act = act_dict[actfun]

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        if self.use_LC and self.rank_num <= 0:
            raise ValueError(
                f'rank_num={rank_num} must be a positive integer when the label corrector is used.')
        
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        if self.use_LC:
            self.rank_feature_emb = nn.Parameter(torch.randn(self.rank_num, self.h_dim))

            if self.LC_classwise:
                self.rank_weight_net = nn.Sequential(
                                    nn.Linear(self.h_dim, self.h_dim),
                                    self.act(),
                                    nn.Linear(self.h_dim, self.h_dim),
                                    self.act(),
                                    nn.Linear(self.h_dim, self.num_classes, bias=True)
                                )
            else:
                self.rank_weight_net = nn.Sequential(
                                    nn.Linear(self.h_dim, self.h_dim),
                                    self.act(),
                                    nn.Linear(self.h_dim, self.h_dim),
                                    self.act(),
                                    nn.Linear(self.h_dim, 1, bias=True)
                                )         

        if self.use_MG:          
            self.cls_feature_emb = nn.Parameter(torch.randn(1, self.h_dim))

            self.margin_generator = nn.Sequential(
                                    nn.Linear(self.h_dim, self.h_dim),
                                    self.act(),
                                    nn.Linear(self.h_dim, self.h_dim),
                                    self.act(),
                                    nn.Linear(self.h_dim, self.num_classes, bias=True) 
                                )
    
    def init_weights(self):
        if self.use_LC:
            nn.init.xavier_uniform_(self.rank_feature_emb)
            nn.init.xavier_normal_(self.rank_weight_net[0].weight)
            nn.init.xavier_normal_(self.rank_weight_net[2].weight)
            nn.init.xavier_normal_(self.rank_weight_net[4].weight)
            self.rank_weight_net[0].bias.data.zero_()
            self.rank_weight_net[2].bias.data.zero_()
            nn.init.constant_(self.rank_weight_net[4].bias, 3.98)


        if self.use_MG:
            nn.init.xavier_uniform_(self.cls_feature_emb)
            nn.init.xavier_normal_(self.margin_generator[0].weight)
            nn.init.xavier_normal_(self.margin_generator[2].weight)
            nn.init.xavier_normal_(self.margin_generator[4].weight)
            self.margin_generator[0].bias.data.zero_()
            self.margin_generator[2].bias.data.zero_()
            # initialize the bias to 1 for preventing the meta train invalid
            nn.init.constant_(self.margin_generator[4].bias, 1)

    def forward_train(self, cls_score, data, warm=False, **kwargs):

        gt_label = data['gt_label']
        gt_label = F.one_hot(gt_label, num_classes=self.num_classes).float()
    
        if self.use_LC:
            data_rank = data['rank']
            data_rank = F.one_hot(data_rank.long(), num_classes=self.rank_num).squeeze(1).float()
            infor = torch.mm(data_rank, self.rank_feature_emb)

            rank_weight = self.rank_weight_net(infor).sigmoid()

            if self.LC_classwise:
                rank_weight = torch.matmul(gt_label.unsqueeze(1), rank_weight.unsqueeze(2)).squeeze(1)

            pred_label = cls_score.detach()
            if warm:
                pred_label = pred_label.softmax(dim=1)
            else:
                with torch.no_grad():
                    pred_label = F.gumbel_softmax(pred_label, tau=1, hard=True)
            correct_label = gt_label * rank_weight + pred_label * (1 - rank_weight)
        else:
            correct_label = gt_label

        if self.use_MG:            
            margin = self.margin_generator(self.cls_feature_emb)
        else:
            margin = torch.zeros((1, self.num_classes)).to(data['gt_label'].device)

        return correct_label, margin

    def get_rank_weight(self):
        data_rank = torch.arange(0, self.rank_num).to(self.rank_weight_net[0].weight.device)
        data_rank = F.one_hot(data_rank.long(), num_classes=self.rank_num).float()
        infor = torch.mm(data_rank, self.rank_feature_emb)
        
        rank_weight = self.rank_weight_net(infor).sigmoid()
        if self.LC_classwise:
            rank_weight = rank_weight.permute(1,0).reshape(-1, 1)

        return rank_weight.squeeze(1).detach().cpu().numpy()
    
    def get_margin(self):
        margin = self.margin_generator(self.cls_feature_emb)
        return margin.squeeze(0).detach().cpu().numpy()
