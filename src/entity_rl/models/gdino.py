# pylint: skip-file
# flake8: noqa

import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmdet.models.detectors import DINO, GroundingDINO
from mmdet.models.detectors.dino import DINO
from mmdet.models.detectors.glip import (
    create_positive_map,
    create_positive_map_label_to_token,
    run_ner,
)
from mmdet.models.layers import SinePositionalEncoding
from mmdet.models.layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder,
    GroundingDinoTransformerEncoder,
)
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import ConfigType
from mmengine.registry import DefaultScope
from torch import Tensor

# This is needed for MM to handle the registry properly
_ = DefaultScope.get_instance("EXPERIMENT", scope_name="mmdet")


class GDino(DINO):
    def __init__(
        self,
        prompt_size,
        max_per_image,
        # language_model,
        *args,
        use_autocast=False,
        **kwargs,
    ) -> None:

        self.prompt_size = prompt_size
        # self.language_model_cfg = language_model
        self._special_tokens = ". "
        self.use_autocast = use_autocast
        self.max_text_len = kwargs["bbox_head"]["contrastive_cfg"]["max_text_len"]
        self.max_per_img = max_per_image
        super().__init__(*args, **kwargs)

        # This is added in DINO.__init__, so it's a bit of a pain to work around it.
        del self.dn_query_generator

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze(model: nn.Module):
        """Freeze the model."""
        model.train()
        for param in model.parameters():
            param.requires_grad = True

    def _init_layers(self) -> None:
        self.cls_features = self.prompt_size

        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            f"embed_dims should be exactly 2 times of num_feats. "
            f"Found {self.embed_dims} and {num_feats}."
        )

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        # self.language_model = MODELS.build(self.language_model_cfg)
        # language_dim = self.language_model.language_backbone.body.language_dim
        # This is fixed by the original BERT language_model.language_backbone.body.language_dim
        language_dim = 768
        self.text_feat_map = nn.Linear(language_dim, self.embed_dims, bias=True)

        # This is the embedded prompt, which shouldn't be longer than the
        # original max_tokens (256)
        self.text_embed = nn.Embedding(self.prompt_size, language_dim)

        # Freeze everything, then only unfreeze the text_embed params
        # This way there's nothing missed (in theory)
        self.freeze(self)
        self.unfreeze(self.text_embed)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

        nn.init.xavier_uniform_(self.text_embed.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ""
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if "prefix" in enhanced_text_dict:
                    caption_string += enhanced_text_dict["prefix"]
                start_i = len(caption_string)
                if "name" in enhanced_text_dict:
                    caption_string += enhanced_text_dict["name"]
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if "suffix" in enhanced_text_dict:
                    caption_string += enhanced_text_dict["suffix"]
            else:
                tokens_positive.append(
                    [[len(caption_string), len(caption_string) + len(word)]]
                )
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ""
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string), len(caption_string) + len(word)]]
            )
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None,
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts
                )
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption
                )

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding="max_length" if self.language_model.pad_to_max else "longest",
                return_tensors="pt",
            )
            entities = original_caption
        else:
            if not original_caption.endswith("."):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding="max_length" if self.language_model.pad_to_max else "longest",
                return_tensors="pt",
            )
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers
            ].max_text_len,
        )
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1
        )
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith("."):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith("."):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding=(
                        "max_length" if self.language_model.pad_to_max else "longest"
                    ),
                    return_tensors="pt",
                )
                positive_map_label_to_token, positive_map = self.get_positive_map(
                    tokenized, tokens_positive
                )

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0] : t[1]])
                    entities.append(" / ".join(instance_entities))
                return (
                    positive_map_label_to_token,
                    original_caption,
                    positive_map,
                    entities,
                )

        chunked_size = self.test_cfg.get("chunked_size", -1)
        if not self.training and chunked_size > 0:
            assert (
                isinstance(original_caption, (list, tuple)) or custom_entities is True
            )
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt
            )
            (
                positive_map_label_to_token,
                caption_string,
                positive_map,
                entities,
            ) = all_output
        else:
            (
                tokenized,
                caption_string,
                tokens_positive,
                entities,
            ) = self.get_tokens_and_prompts(
                original_caption, custom_entities, enhanced_text_prompt
            )
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive
            )
        return positive_map_label_to_token, caption_string, positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
        self,
        original_caption: Union[list, tuple],
        enhanced_text_prompts: Optional[ConfigType] = None,
    ):
        chunked_size = self.test_cfg.get("chunked_size", -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(list(range(1, len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts
                )
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i]
                )
            tokenized = self.language_model.tokenizer(
                [caption_string], return_tensors="pt"
            )
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn(
                    "Inputting a text that is too long will result "
                    "in poor prediction performance. "
                    "Please reduce the --chunked-size."
                )
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive
            )

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return (
            positive_map_label_to_token_chunked,
            caption_string_chunked,
            positive_map_chunked,
            entities_chunked,
        )

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples
        )

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict
        )

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples
        )
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(
        self,
        feat: Tensor,
        feat_mask: Tensor,
        feat_pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        text_dict: Dict,
    ) -> Dict:
        text_token_mask = text_dict["text_token_mask"]
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict["embedded"],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["masks"],
        )
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
        )
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )

        enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](
            output_memory, memory_text, text_token_mask
        )
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ].max_text_len
        enc_outputs_coord_unact = (
            self.bbox_head.reg_branches[self.decoder.num_layers](output_memory)
            + output_proposals
        )

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        top_k = min(self.num_queries, enc_outputs_class.shape[1])
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], k=top_k, dim=1)[1]
        # import ipdb; ipdb.set_trace()

        topk_score = torch.gather(
            enc_outputs_class,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features),
        )
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4)
        )
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        # NOTE: the actual number of proposals can be lower than the
        # number of queries for small images, so we need to select
        query = []
        for i in range(bs):
            query.append(self.query_embedding.weight[topk_indices[i]])

        query = torch.stack(query, dim=0)

        # query = self.query_embedding.weight[:, None, :]
        # query = query.repeat(1, bs, 1).transpose(0, 1)

        # We don't need the DN queries for ENROS
        # TODO: clean this up
        # if self.training:
        #     dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(
        #         batch_data_samples
        #     )
        #     query = torch.cat([dn_label_query, query], dim=1)
        #     reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
        # else:
        #     reference_points = topk_coords_unact
        #     dn_mask, dn_meta = None, None

        reference_points = topk_coords_unact
        dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = (
            dict(
                enc_outputs_class=topk_score,
                enc_outputs_coord=topk_coords,
                dn_meta=dn_meta,
            )
            if self.training
            else dict()
        )
        # append text_feats to head_inputs_dict
        head_inputs_dict["memory_text"] = memory_text
        head_inputs_dict["text_token_mask"] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        text_prompts = [data_samples.text for data_samples in batch_data_samples]

        gt_labels = [
            data_samples.gt_instances.labels for data_samples in batch_data_samples
        ]

        if "tokens_positive" in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                tokens_positive, text_prompts, gt_labels
            ):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding=(
                        "max_length" if self.language_model.pad_to_max else "longest"
                    ),
                    return_tensors="pt",
                )
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                (
                    tokenized,
                    caption_string,
                    tokens_positive,
                    _,
                ) = self.get_tokens_and_prompts(text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [tokens_positive[label] for label in gt_label]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive
                    )
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    (
                        tokenized,
                        caption_string,
                        tokens_positive,
                        _,
                    ) = self.get_tokens_and_prompts(text_prompt, True)
                    new_tokens_positive = [tokens_positive[label] for label in gt_label]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive
                    )
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(batch_inputs.device).bool().float()
            text_token_mask = text_dict["text_token_mask"][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = text_token_mask.unsqueeze(
                0
            ).repeat(len(positive_map), 1)
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            visual_features, text_dict, batch_data_samples
        )

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples
        )
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if "caption_prompt" in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get("tokens_positive", None))

        if "custom_entities" in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0],
                    custom_entities,
                    enhanced_text_prompts[0],
                    tokens_positives[0],
                )
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompt, custom_entities, enhanced_text_prompt, tokens_positive
                )
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives
                )
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts
        )

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])

                batch_data_samples[0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples
                )
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples,
                )[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples
            )
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples,
            )

        for data_sample, pred_instances, entity, is_rec_task in zip(
            batch_data_samples, results_list, entities, is_rec_tasks
        ):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            "The unexpected output indicates an issue with "
                            "named entity recognition. You can try "
                            "setting custom_entities=True and running "
                            "again to see if it helps."
                        )
                        label_names.append("unobject")
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples

    def _forward(self, batch_inputs, batch_data_samples):
        return self.__forward(batch_inputs, batch_data_samples)

    def __forward(self, batch_inputs, batch_data_samples):
        if batch_data_samples is None:
            batch_data_samples = []
            for inputs in batch_inputs:
                # Shape for each input is (C, H, W)
                # The batch_input_shape is (H, W)
                # And we also need img_shape
                batch_data_samples.append(
                    DetDataSample(
                        batch_input_shape=inputs.shape[1:],
                        img_shape=inputs.shape[1:],
                    )
                )

        # Orig:
        # text_dict = self.language_model(list(text_prompts))

        batch_size = len(batch_inputs)

        embedded_text = self.text_embed.weight
        embedded_text = self.text_feat_map(embedded_text)

        # Zero-pad the embedded_text to the max_text_len
        # pad_length = self.max_text_len - self.prompt_size
        # embedded_text = torch.cat(
        #     [
        #         embedded_text,
        #         torch.zeros(pad_length, embedded_text.shape[-1], device=self.device),
        #     ],
        #     dim=0,
        # )

        # The token mas is all true, except for the padding
        # text_token_mask = torch.cat(
        #     [
        #         torch.ones(self.prompt_size, dtype=torch.bool, device=self.device),
        #         torch.zeros(pad_length, dtype=torch.bool, device=self.device),
        #     ],
        #     dim=0,
        # )
        text_token_mask = torch.ones(
            self.prompt_size,
            dtype=torch.bool,
            device=self.device,
        )

        # TODO: verify the special tokens at this point
        # TODO: move this to init
        text_dict = {
            # Only attend to the individual tokens (for now)
            # With sub_sentence_represent, the model will attend to the
            # blocks within the sentence.
            "masks": torch.eye(
                self.prompt_size,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0),
            # Same as above, position ids are 0 for all, because they
            # are treated as a single sentence.
            "position_ids": torch.zeros(
                self.prompt_size,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0),
            "text_token_mask": text_token_mask.unsqueeze(0),
            # This is the learnt embedding of the text prompt, but for
            # efficiency, text_embed is used to represent the values.
            "embedded": embedded_text.unsqueeze(0),
        }

        for k in text_dict:
            non_batch_dims = [-1] * (text_dict[k].dim() - 1)
            text_dict[k] = text_dict[k].expand(batch_size, *non_batch_dims)
            # text_dict[k] = torch.cat(batch_size * [text_dict[k]])

        img_feats = self.extract_feat(batch_inputs)

        head_inputs_dict = self.forward_transformer(
            img_feats, text_dict=text_dict, batch_data_samples=batch_data_samples
        )

        all_layers_cls_scores, all_layers_bbox_preds = self.bbox_head.forward(
            hidden_states=head_inputs_dict["hidden_states"],
            references=head_inputs_dict["references"],
            memory_text=head_inputs_dict["memory_text"],
            text_token_mask=head_inputs_dict["text_token_mask"],
        )

        # TODO: add token_positive_map to align with the original
        # prompt input

        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]

        # This is just filtering out all the -inf scores
        cls_scores = cls_scores[:, :, : self.prompt_size]

        results = []
        # TODO: this will return all queries, not just the top-k
        for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
            results.append(self._predict_single(cls_score, bbox_pred, self.max_per_img))

        results_batched = {}
        for k in results[0]:
            results_batched[k] = torch.stack([r[k] for r in results])

        return results_batched

    def _predict_single(self, cls_score, bbox_pred, max_per_img=None):
        cls_score = cls_score.sigmoid()
        scores, _ = cls_score.max(-1)

        if max_per_img is None:
            max_per_img = self.num_queries

        max_per_img = min(scores.shape[0], max_per_img)
        scores, indexes = scores.topk(max_per_img)
        bbox_pred = bbox_pred[indexes]
        cls_features = cls_score[indexes]
        # det_labels = scores.new_zeros(scores.shape, dtype=torch.long)

        det_bboxes = bbox_pred
        # det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        # det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        # det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        # det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        # det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        # if rescale:
        #     assert img_meta.get("scale_factor") is not None
        #     det_bboxes /= det_bboxes.new_tensor(img_meta["scale_factor"]).repeat((1, 2))
        # results.bboxes = det_bboxes
        # results.scores = scores
        # results.labels = det_labels
        return {
            "bboxes": det_bboxes,
            "scores": scores,
            "features": cls_features,
            # "labels": det_labels,
        }
