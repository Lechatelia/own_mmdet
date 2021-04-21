import torch.nn as nn

from ..registry import HEADS
from mmdet.ops import ConvModule
from .bbox_head import BBoxHead


@HEADS.register_module
class GCNBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 nongt_dim=128,
                 num_rois = 256,
                 dropout = 0.5,
                 *args,
                 **kwargs):
        super(GCNBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.nongt_dim = nongt_dim
        self.num_rois = num_rois
        self.dropout = dropout

        self.relation_module = nn.ModuleList()
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append( # 添加非线性函数和dropout
                    nn.Sequential(nn.Linear(fc_in_channels, self.fc_out_channels),
                                  nn.ReLU())
                                  # nn.Dropout(self.dropout))
                    )
                self.relation_module.append(Relation_Encoder(v_dim=self.fc_out_channels, out_dim=self.fc_out_channels,
                                                   nongt_dim=self.nongt_dim, dir_num=1, pos_emb_dim=64, num_steps=1)
                                   )
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(GCNBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, rois, bs):
        # shared part x[512*bs, 256, 7, 7]
        pos_emb = prepare_graph_variables(rois.view(bs, int(rois.size(0)/bs),-1), nongt_dim=self.nongt_dim, device=rois.device)
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1).view(bs, int(x.size(0)/bs), -1)

            for i, fc in enumerate(self.shared_fcs): # 两个全连接网络
                x = self.relu(fc(x))
                x = self.relation_module[i](x, pos_emb)
        # separate branches
        x = x.view(-1, x.size(-1))
        x_cls = x # [bs*num_rois, 1024]
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None #分别进行分类
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None # 位置回归
        return cls_score, bbox_pred


from torch.nn.utils.weight_norm import weight_norm
import torch
from torch.autograd import Variable
import math
class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='ReLU', dropout=0, bias=True):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim, bias=bias),
                                      dim=None))
            if '' != act and act is not None:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1], bias=bias),
                                  dim=None))
        if '' != act and act is not None:
            layers.append(getattr(nn, act)())


        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class GraphSelfAttentionLayer(nn.Module):
    def __init__(self, feat_dim, nongt_dim=20, pos_emb_dim=-1,
                 num_heads=16, dropout=[0.2, 0.2]):
                 # num_heads=16, dropout=[0.2, 0.5]):
        """ Attetion module with vectorized version

        Args:
            position_embedding: [num_rois, nongt_dim, pos_emb_dim]
                                used in implicit relation
            pos_emb_dim: set as -1 if explicit relation
            nongt_dim: number of objects consider relations per image
            fc_dim: should be same as num_heads
            feat_dim: dimension of roi_feat
            num_heads: number of attention heads
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        super(GraphSelfAttentionLayer, self).__init__()
        # multi head
        self.fc_dim = num_heads
        self.feat_dim = feat_dim
        self.dim = (feat_dim, feat_dim, feat_dim)
        self.dim_group = (int(self.dim[0] / num_heads),
                          int(self.dim[1] / num_heads),
                          int(self.dim[2] / num_heads))
        self.num_heads = num_heads
        self.pos_emb_dim = pos_emb_dim
        if self.pos_emb_dim > 0:
            self.pair_pos_fc1 = FCNet([pos_emb_dim, self.fc_dim], None, dropout[0]) # --16
        self.query = FCNet([feat_dim, self.dim[0]], None, dropout[1]) # 1024
        self.nongt_dim = nongt_dim

        self.key = FCNet([feat_dim, self.dim[1]], None, dropout[1])

        self.linear_out_ = weight_norm(
                            nn.Conv2d(in_channels=self.fc_dim * feat_dim,
                                      out_channels=self.dim[2],
                                      kernel_size=(1, 1),
                                      groups=self.fc_dim), dim=None) # 每一个num_heads通道之间进行group卷积，

        self.activ = nn.ReLU()

    def forward(self, roi_feat, adj_matrix,
                position_embedding, label_biases_att):
        """
        Args:
            roi_feat: [batch_size, N, feat_dim]
            adj_matrix: [batch_size, N, nongt_dim]
            position_embedding: [batch_size, num_rois, nongt_dim, pos_emb_dim]
            label_biases_att: [batch_size, N, nongt_dim]
        Returns:
            output: [batch_size, num_rois, ovr_feat_dim, output_dim]
        """
        batch_size = roi_feat.size(0)
        num_rois = roi_feat.size(1)
        # nongt_dim: number of objects consider relations per image
        nongt_dim = self.nongt_dim if self.nongt_dim < num_rois else num_rois
        # [batch_size,nongt_dim, feat_dim]
        nongt_roi_feat = roi_feat[:, :nongt_dim, :] # 只考虑到 nongt_dim 个proposal的特征

        # [batch_size,num_rois, self.dim[0] = feat_dim]
        q_data = self.query(roi_feat)

        # [batch_size,num_rois, num_heads, feat_dim /num_heads]
        q_data_batch = q_data.view(batch_size, num_rois, self.num_heads,
                                   self.dim_group[0])

        # [batch_size,num_heads, num_rois, feat_dim /num_heads]
        q_data_batch = torch.transpose(q_data_batch, 1, 2)

        # [batch_size,nongt_dim, self.dim[1] = feat_dim]
        k_data = self.key(nongt_roi_feat)

        # [batch_size,nongt_dim, num_heads, feat_dim /num_heads]
        k_data_batch = k_data.view(batch_size, nongt_dim, self.num_heads,
                                   self.dim_group[1])

        # [batch_size,num_heads, nongt_dim, feat_dim /num_heads]
        k_data_batch = torch.transpose(k_data_batch, 1, 2)

        # [batch_size,nongt_dim, feat_dim]
        v_data = nongt_roi_feat

        # [batch_size,num_heads, num_rois, feat_dim /num_heads] *  # [batch_size,num_heads, nongt_dim, feat_dim /num_heads]
        # [batch_size, num_heads, num_rois, nongt_dim]
        aff = torch.matmul(q_data_batch, torch.transpose(k_data_batch, 2, 3))

        # aff_scale, [batch_size, num_heads, num_rois, nongt_dim]
        aff_scale = (1.0 / math.sqrt(float(self.dim_group[1]))) * aff #相当于除以sqrt(d_k) 相当于归一化
        # aff_scale, [batch_size,num_rois,num_heads, nongt_dim]
        aff_scale = torch.transpose(aff_scale, 1, 2)
        weighted_aff = aff_scale # 相当于只考虑视觉特征的关系矩阵

        if position_embedding is not None and self.pos_emb_dim > 0:
            # Adding goemetric features
            position_embedding = position_embedding.float()
            # [batch_size,num_rois * nongt_dim, emb_dim]
            position_embedding_reshape = position_embedding.view(
                (batch_size, -1, self.pos_emb_dim))

            # position_feat_1, [batch_size,num_rois * nongt_dim, fc_dim]
            position_feat_1 = self.pair_pos_fc1(position_embedding_reshape) #维度变化  [batch_size,num_rois * nongt_dim, num_heads]
            position_feat_1_relu = nn.functional.relu(position_feat_1)

            # aff_weight, [batch_size,num_rois, nongt_dim, fc_dim]
            aff_weight = position_feat_1_relu.view(
                (batch_size, -1, nongt_dim, self.fc_dim))

            # aff_weight, [batch_size,num_rois, fc_dim, nongt_dim]
            aff_weight = torch.transpose(aff_weight, 2, 3)

            thresh = torch.FloatTensor([1e-6]).cuda()
            # weighted_aff, [batch_size,num_rois, fc_dim, nongt_dim]
            threshold_aff = torch.max(aff_weight, thresh) # 相当于一个clip操作

            weighted_aff += torch.log(threshold_aff) # 视觉关系矩阵加上位置关系矩阵

        # 因为adj_matrix 是全1矩阵，所以这部分代码没有意义
        if adj_matrix is not None:
            # weighted_aff_transposed, [batch_size,num_rois, nongt_dim, num_heads]
            weighted_aff_transposed = torch.transpose(weighted_aff, 2, 3)
            zero_vec = -9e15*torch.ones_like(weighted_aff_transposed)

            adj_matrix = adj_matrix.view(
                            adj_matrix.shape[0], adj_matrix.shape[1],
                            adj_matrix.shape[2], 1) #[bs, N, nongt_num. 1]
            adj_matrix_expand = adj_matrix.expand(
                                (-1, -1, -1,
                                 weighted_aff_transposed.shape[-1])) # [bs, N, nongt_num. num_heads]
            weighted_aff_masked = torch.where(adj_matrix_expand > 0,
                                              weighted_aff_transposed,
                                              zero_vec) # 根据开始求得邻接矩阵进行选择，如果大于0的就是计算的系数，如果不大于0就用0填充
# 这里为什么要加上label biases呢，就是因为如果邻接矩阵中如果本来存在连接，就加上一个bias，使得更高，但我感觉也没啥意义，
            weighted_aff_masked = weighted_aff_masked + label_biases_att.unsqueeze(3) # [bs, N, nongt_num, num_heads]
            weighted_aff = torch.transpose(weighted_aff_masked, 2, 3)  # [bs, N,  num_heads， nongt_num]

        # aff_softmax, [batch_size, num_rois, fc_dim, nongt_dim]
        aff_softmax = nn.functional.softmax(weighted_aff, 3) #对输入的特征做一个softmax进行归一化

        # aff_softmax_reshape, [batch_size, num_rois*fc_dim, nongt_dim]
        aff_softmax_reshape = aff_softmax.view((batch_size, -1, nongt_dim))

        # output_t, [batch_size, num_rois*fc_dim, nongt_dim]*[batch_size,nongt_dim, feat_dim]--》
        # [batch_size, num_rois * fc_dim, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data) #将这 nongt_dim个proposal的特征进行聚合操作

        # output_t, [batch_size*num_rois, fc_dim * feat_dim, 1, 1]
        output_t = output_t.view((-1, self.fc_dim * self.feat_dim, 1, 1))

        # linear_out, [batch_size*num_rois, dim[2], 1, 1]
        linear_out = self.linear_out_(output_t) # 利用二维卷积的group卷积操作实现不同heads的信息聚合
        output = linear_out.view((batch_size, num_rois, self.dim[2])) #[bs, num_roi, feat_dim]
        # todo add relu function

        return self.activ(output)


class GAttNet(nn.Module):
    def __init__(self, dir_num, label_num, in_feat_dim, out_feat_dim,
                 nongt_dim=20, dropout=0.7, label_bias=True,
                 num_heads=16, pos_emb_dim=-1):
        """ Attetion module with vectorized version

        Args:
            label_num: numer of edge labels
            dir_num: number of edge directions
            feat_dim: dimension of roi_feat
            pos_emb_dim: dimension of postion embedding for implicit relation, set as -1 for explicit relation

        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        super(GAttNet, self).__init__()
        assert dir_num <= 2, "Got more than two directions in a graph."
        self.dir_num = dir_num #图结构的方向数
        self.label_num = label_num # numer of edge labels 对spatial edge 和 semantic edge进行了类别的划分
        self.in_feat_dim = in_feat_dim  # dimension of input roi_feat
        self.out_feat_dim = out_feat_dim # dimension of output roi_feat
        self.dropout = nn.Dropout(dropout)
        self.self_weights = FCNet([in_feat_dim, out_feat_dim], '', dropout)
        self.label_num = label_num
        if label_num >1:
            self.bias = FCNet([label_num, 1], '', 0, label_bias)
        else:
            self.bias = None
        self.nongt_dim = nongt_dim
        self.pos_emb_dim = pos_emb_dim # position embedding 的维度大小
        neighbor_net = []
        for i in range(dir_num): # 对每一个方向， 建立一个不同的自注意力机制的图结构
            g_att_layer = GraphSelfAttentionLayer(
                                pos_emb_dim=pos_emb_dim,
                                num_heads=num_heads,
                                feat_dim=out_feat_dim,
                                nongt_dim=nongt_dim)
            neighbor_net.append(g_att_layer)
        self.neighbor_net = nn.ModuleList(neighbor_net)

    def forward(self, v_feat, adj_matrix=None, pos_emb=None):
        """
        Args:
            v_feat: [batch_size,num_rois, feat_dim]
            adj_matrix: [batch_size, num_rois, num_rois, num_labels]
            pos_emb: [batch_size, num_rois, pos_emb_dim]

        Returns:
            output: [batch_size, num_rois, feat_dim]
        """
        if self.pos_emb_dim > 0 and pos_emb is None:
            raise ValueError(
                f"position embedding is set to None "
                f"with pos_emb_dim {self.pos_emb_dim}")
        elif self.pos_emb_dim < 0 and pos_emb is not None:
            raise ValueError(
                f"position embedding is NOT None "
                f"with pos_emb_dim < 0")
        batch_size, num_rois, feat_dim = v_feat.shape
        nongt_dim = self.nongt_dim

        if adj_matrix is not None:
            adj_matrix = adj_matrix.float() #  [batch_size, num_rois, num_rois, 1]

            adj_matrix_list = [adj_matrix, adj_matrix.transpose(1, 2)] # 交换一下主次关系， 如果是有两个方向的图结构的话

        # Self - looping edges
        # [batch_size,num_rois, out_feat_dim]
        self_feat = self.self_weights(v_feat) # 相当于GCN当中的W矩阵

        output = self_feat
        neighbor_emb = [0] * self.dir_num
        for d in range(self.dir_num): #对于每个方向的图结构进行运算
            # [batch_size,num_rois, nongt_dim, label_num] --》[ba, 256, 128, 1]
            if adj_matrix is not None:
                input_adj_matrix = adj_matrix_list[d][:, :, :nongt_dim, :]
                if self.label_num > 1:
                    condensed_adj_matrix = torch.sum(input_adj_matrix, dim=-1) ## 在edge label维度上面进行求和 [batch_size,num_rois, nongt_dim]

                    # # [batch_size,num_rois, nongt_dim, label_num]--》 [batch_size,num_rois, nongt_dim]
                    v_biases_neighbors = self.bias(input_adj_matrix).squeeze(-1) #在label维度上面进行全连接， # [batch_size,num_rois, nongt_dim, 1]
                else:
                    condensed_adj_matrix = input_adj_matrix.squeeze(-1)
                    v_biases_neighbors = condensed_adj_matrix
                    # [batch_size,num_rois, out_feat_dim]
                neighbor_emb[d] = self.neighbor_net[d].forward(
                    self_feat, condensed_adj_matrix, pos_emb,v_biases_neighbors)

            else:
                neighbor_emb[d] = self.neighbor_net[d].forward(
                    self_feat, None, pos_emb, None)

            # [batch_size,num_rois, out_feat_dim]
            output = output + neighbor_emb[d] #聚合来的特征加上原来的特征

            #这样的图卷积公式相当于从 A*X*W  ----》》》 (1+A)*X*W
        output = self.dropout(output)
        output = nn.functional.relu(output)

        return output


def torch_extract_position_embedding(position_mat, feat_dim, wave_length=1000,
                                     device=torch.device("cuda")):
    # position_mat, [batch_size,num_rois, nongt_dim, 4]
    feat_range = torch.arange(0, feat_dim / 8)
    dim_mat = torch.pow(torch.ones((1,)) * wave_length,
                        (8. / feat_dim) * feat_range)
    dim_mat = dim_mat.view(1, 1, 1, -1).to(device)
    position_mat = torch.unsqueeze(100.0 * position_mat, dim=4)
    div_mat = torch.div(position_mat.to(device), dim_mat)
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    # embedding, [batch_size,num_rois, nongt_dim, 4, feat_dim/4]
    embedding = torch.cat([sin_mat, cos_mat], -1)
    # embedding, [batch_size,num_rois, nongt_dim, feat_dim]
    embedding = embedding.view(embedding.shape[0], embedding.shape[1],
                               embedding.shape[2], feat_dim)
    return embedding


def torch_extract_position_matrix(bbox, nongt_dim=36):
    """ Extract position matrix

    Args:
        bbox: [batch_size, num_boxes, 4]

    Returns:
        position_matrix: [batch_size, num_boxes, nongt_dim, 4]
    """

    xmin, ymin, xmax, ymax = torch.split(bbox, 1, dim=-1)
    # [batch_size,num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [batch_size,num_boxes, num_boxes]
    delta_x = center_x - torch.transpose(center_x, 1, 2)
    delta_x = torch.div(delta_x, bbox_width)

    delta_x = torch.abs(delta_x)
    threshold = 1e-3
    delta_x[delta_x < threshold] = threshold
    delta_x = torch.log(delta_x)
    delta_y = center_y - torch.transpose(center_y, 1, 2)
    delta_y = torch.div(delta_y, bbox_height)
    delta_y = torch.abs(delta_y)
    delta_y[delta_y < threshold] = threshold
    delta_y = torch.log(delta_y)
    delta_width = torch.div(bbox_width, torch.transpose(bbox_width, 1, 2))
    delta_width = torch.log(delta_width)
    delta_height = torch.div(bbox_height, torch.transpose(bbox_height, 1, 2))
    delta_height = torch.log(delta_height)
    concat_list = [delta_x, delta_y, delta_width, delta_height]
    for idx, sym in enumerate(concat_list):
        # [batch_size, nongt_dim, num_boxes]
        sym = sym[:, :nongt_dim]
        concat_list[idx] = torch.unsqueeze(sym, dim=3)
    position_matrix = torch.cat(concat_list, 3)
    return position_matrix


class Relation_Encoder(nn.Module):
    def __init__(self, v_dim, out_dim, nongt_dim, dir_num=1, pos_emb_dim=64,
                 num_heads=16, num_steps=1,
                 residual_connection=True, label_bias=True):
        # nongt_dim number of objects consider relations per image
        super(Relation_Encoder, self).__init__()
        self.v_dim = v_dim  # 特征维度
        self.out_dim = out_dim  # 输出维度
        self.residual_connection = residual_connection  # 是否采用残差结构
        self.num_steps = num_steps  # 进行多少次GCN
        print("In ImplicitRelationEncoder, num of graph propogate steps:",
              "%d, residual_connection: %s" % (self.num_steps,
                                               self.residual_connection))

        if self.v_dim != self.out_dim:
            self.v_transform = FCNet([v_dim, out_dim])  # 如果维度不一致，就先改编维度
        else:
            self.v_transform = None
        self.implicit_relation = GAttNet(dir_num, 1, out_dim, out_dim,
                                     nongt_dim=nongt_dim,
                                     label_bias=label_bias,
                                     num_heads=num_heads,
                                     pos_emb_dim=pos_emb_dim)

    def forward(self, v, position_embedding):
        """
        Args:
            v: [batch_size, num_rois, v_dim]  视觉特征
            position_embedding: [batch_size, num_rois, nongt_dim, emb_dim]

        Returns:
            output: [batch_size, num_rois, out_dim,3]
        """
        # [batch_size, num_rois, num_rois, 1]
        imp_adj_mat = Variable(torch.ones(v.size(0), v.size(1), v.size(1), 1)).to(v.device)
        # 全1矩阵 [batch_szie, 256, 256, 1]     其实并没有什么意义 所以这里改成 by zjg 20200314
        # imp_adj_mat = None

        imp_v = self.v_transform(v) if self.v_transform else v  # 先检查特征维度

        for i in range(self.num_steps):
            imp_v_rel = self.implicit_relation.forward(imp_v,
                                                       imp_adj_mat,
                                                       position_embedding)
            if self.residual_connection:
                output = imp_v + imp_v_rel
            else:
                output = imp_v_rel
        return output


def prepare_graph_variables(bb, nongt_dim, device, pos_emb_dim=64):
    # bbox: [batch_size, num_boxes, 4]
    # pos_emd_dim position_embedding_dim:

    # bb = bb  # [batch_size, num_boxes, 4]
    pos_mat = torch_extract_position_matrix(bb, nongt_dim=nongt_dim)  # [batch_size, num_boxes, nongt_dim, 4]
    pos_emb = torch_extract_position_embedding(pos_mat, feat_dim=pos_emb_dim, device=device)
    # position embedding, [batch_size,num_rois, nongt_dim, feat_dim]
    pos_emb_var = Variable(pos_emb)
    return pos_emb_var


def test_GCN():
    print('CUDA available: {}'.format(torch.cuda.is_available()))
    print('the available CUDA number is : {}'.format(torch.cuda.device_count()))
    nongt_dim = 128
    rois = torch.randn(5, 256, 4).cuda()
    pooled_feat = torch.randn(5, 256, 49).cuda()
    relation_encode = Relation_Encoder(v_dim=49, out_dim=1024, nongt_dim=nongt_dim, dir_num=1, pos_emb_dim=64,
                                       num_steps=3)
    relation_encode = nn.DataParallel(relation_encode.cuda())
    pos_emb = prepare_graph_variables(rois, nongt_dim=nongt_dim, device=rois.device)
    realtion_feat = relation_encode(pooled_feat, pos_emb)
    print(realtion_feat.size())
