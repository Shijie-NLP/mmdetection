# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import OptSampleList, DetDataSample
from mmdet.utils import OptConfigType, ConfigType
from ..layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder,
                      DinoTransformerDecoder, SinePositionalEncoding)
from ..utils import multi_apply
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention


@MODELS.register_module()
class MyDINO(DeformableDETR):
    r"""Implementation of `DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection <https://arxiv.org/abs/2203.03605>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        dn_cfg (:obj:`ConfigDict` or dict, optional): Config of denoising
            query generator. Defaults to `None`.
    """

    def __init__(self, *args, dn_cfg: OptConfigType = None,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 topk: int = 9, learned_query: bool = True, **kwargs) -> None:
        self.topk = topk
        self.learned_query = learned_query
        super().__init__(*args, **kwargs)

        assert self.as_two_stage, 'as_two_stage must be True for DINO'
        assert self.with_box_refine, 'with_box_refine must be True for DINO'

        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = self.bbox_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
        self.dn_query_generator = CdnQueryGenerator(**dn_cfg)

        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        if self.learned_query:
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory)

        enc_outputs_coord_unact = self.bbox_head.reg_branches[
                                      self.decoder.num_layers](output_memory) + output_proposals

        num_queries_list = [self.num_queries] * bs
        topk_list = [self.topk] * bs  # 这里的 topk 是你传给 _get_topk_single 的参数

        all_scores_list, all_bboxes_unact_list, all_attn_mask_list = multi_apply(
            self._get_topk_single,
            enc_outputs_class,
            enc_outputs_coord_unact,
            batch_data_samples if self.training else [None] * bs,
            num_queries_list,
            topk_list,
        )

        topk_score = torch.stack(all_scores_list, dim=0)
        topk_coords_unact = torch.stack(all_bboxes_unact_list, dim=0)
        decoder_attn_mask = all_attn_mask_list[0]

        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact_for_ref = topk_coords_unact.detach()

        if self.learned_query:
            query = self.query_embedding.weight[:, None, :]
            query = query.repeat(1, bs, 1).transpose(0, 1)
        else:
            raise NotImplementedError

        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)

            num_denoising_queries = dn_meta["num_denoising_queries"]
            dn_mask[num_denoising_queries:, num_denoising_queries:] = decoder_attn_mask

            query = torch.cat([dn_label_query, query], dim=1)
            # 将 DN 的 bbox 和我们筛选出的 proposal 拼接作为初始参考点
            reference_points_unact = torch.cat([dn_bbox_query, topk_coords_unact_for_ref],
                                               dim=1)
        else:
            reference_points_unact = topk_coords_unact_for_ref
            dn_mask, dn_meta = decoder_attn_mask, None

        reference_points = reference_points_unact.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)

        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()

        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict

    def _get_topk_single(self,
                         pred_scores: Tensor,
                         preds_bboxes_unact: Tensor,
                         data_samples: DetDataSample,
                         num_queries: int,
                         topk: int,
                         giou_weight: float = 2.0,
                         dist_weight: float = 0.5
                         ) -> tuple:
        valid_mask = preds_bboxes_unact.ne(torch.inf).all(dim=-1)
        safe_unact_bboxes = preds_bboxes_unact[valid_mask]
        safe_scores = pred_scores[valid_mask]

        # --- 2. 计算排序基准 (Quality Score) ---
        if data_samples is not None and len(data_samples.gt_instances.bboxes) > 0:
            img_h, img_w = data_samples.img_shape
            scale_factor = preds_bboxes_unact.new_tensor([img_w, img_h, img_w, img_h])

            preds_act = safe_unact_bboxes.sigmoid()
            cx, cy, w, h = preds_act.unbind(-1)
            preds_xyxy_norm = torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

            # 【训练模式】计算 Fused Score (GIOU - Distance)
            gt_bboxes_norm = data_samples.gt_instances.bboxes / scale_factor
            giou_scores = self.iou_calculator(preds_xyxy_norm, gt_bboxes_norm, mode="giou")

            gt_centers_norm = (gt_bboxes_norm[:, :2] + gt_bboxes_norm[:, 2:]) / 2.0
            dists = torch.cdist(preds_act[:, :2], gt_centers_norm, p=2)

            # 每个 Query 选它匹配得最好的那个 GT 的分数作为自己的质量分
            matching_matrix = (giou_weight * giou_scores) - (dist_weight * dists)
            quality_score, _ = matching_matrix.max(dim=1)

            # 选出必须包含的 TopK 索引
            real_topk = min(topk, safe_unact_bboxes.size(0))
            _, topk_indices_per_gt = matching_matrix.topk(real_topk, dim=0)
            mandatory_indices = topk_indices_per_gt.flatten().unique()
        else:
            # 【推理模式】没有 GT，直接用分类分数作为质量分
            quality_score, _ = safe_scores.max(dim=-1)
            mandatory_indices = torch.tensor([], dtype=torch.long, device=pred_scores.device)

        # --- 3. 筛选与补齐 ---
        max_cls_scores, _ = safe_scores.max(dim=-1)
        candidate_scores = max_cls_scores.clone()
        candidate_scores[mandatory_indices] = -1e9

        target_num = min(num_queries, safe_unact_bboxes.size(0))
        num_needed = target_num - mandatory_indices.numel()

        if num_needed > 0:
            _, remaining_indices = candidate_scores.topk(num_needed)
            final_indices = torch.cat([mandatory_indices, remaining_indices])
        else:
            final_indices = mandatory_indices[:target_num]

        # --- 4. 排序并生成 Mask ---
        # 关键：按照质量分降序排列，确保“优等生”排在前面
        selected_quality = quality_score[final_indices]
        _, sort_idx = selected_quality.sort(descending=True)
        final_indices = final_indices[sort_idx]

        # 生成单向注意力掩码：高分不看低分
        N = final_indices.numel()
        attn_mask = torch.zeros((N, N), device=pred_scores.device, dtype=torch.bool)
        # triu(diagonal=1) 得到严格上三角矩阵，这些位置设为 -inf 表示不可见
        mask_indices = torch.triu(torch.ones(N, N, device=pred_scores.device), diagonal=1).bool()
        attn_mask[mask_indices] = True

        return safe_scores[final_indices], safe_unact_bboxes[final_indices], attn_mask
