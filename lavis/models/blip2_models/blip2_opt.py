"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
# from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig
import transformers

# SAM
from transformers import SamModel, SamProcessor
from PIL import Image
from torchvision.transforms import ToPILImage
# from visualizer import get_local
# END SAM



@registry.register_model("blip2_opt")
class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"
        
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
            print("----------------------------------------------opt---------------------------------------------------------------")

        # for name, param in self.Qformer.named_parameters():
        #     param.requires_grad = False
        # self.query_tokens.requires_grad = False

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        # prepared for opt?
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        # 

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        # SAM
            # 1. SAM's Qformer
        self.sam_Qformer, self.sam_query_tokens = self.init_sam_Qformer(
            num_query_token, 4096
        )
                # prepared for opt
        self.sam_Qformer.cls = None
        self.sam_Qformer.bert.embeddings.word_embeddings = None
        self.sam_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.sam_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
            # 2. SAM's Model
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-big")
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-big")
            # 3. SAM's Projection Head
        self.sam_opt_proj = nn.Linear(
            self.sam_Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )
        self.output_dict = {}
        def hook_fn(module, input, output, name):
            self.output_dict[name] = output.detach().cpu()


        self.sam_Qformer.bert.encoder.layer[10].crossattention.self.query.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'query'))
        self.sam_Qformer.bert.encoder.layer[10].crossattention.self.key.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'key'))
        if freeze_vit:
            for name, param in self.sam_model.named_parameters():
                param.requires_grad = False
            self.sam_model = self.sam_model.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze SAM encoder")
        # END SAM    

    def forward(self, samples):
        # print("forward, hahahaha")
        image = samples["image"]
        img_per_sample = image.shape[1]
        image_atts = []
        image_embeds = []
        sam_embeds = []
        sam_atts = []
        for i in range(img_per_sample):
            with self.maybe_autocast():
                tmp_image = image[:, i, :, :, :]
                image_embeds_tmp = self.ln_vision(self.visual_encoder(tmp_image))
                # SAM
                to_pil = ToPILImage()
                sam_inputs = None
                
                for index in range(tmp_image.shape[0]):
                    sam_image = to_pil(tmp_image[index])
                    sam_input = self.sam_processor(sam_image, return_tensors="pt").to(image.device)
                    if index==0:
                        sam_inputs = sam_input["pixel_values"]
                    else:
                        sam_inputs = torch.cat([sam_inputs, sam_input["pixel_values"]], dim=0)
                sam_image_embeddings = self.sam_model.get_image_embeddings(sam_inputs).detach()  
                sam_embeddings_shape = sam_image_embeddings.shape
                sam_embeds_tmp = sam_image_embeddings.view(sam_embeddings_shape[0], sam_embeddings_shape[1], -1)
                sam_embeds_tmp = sam_embeds_tmp.to(image.device)

            sam_atts_tmp = torch.ones(sam_embeds_tmp.size()[:-1], dtype=torch.long).to(
                image.device
            )
            # END SAM
            image_atts_tmp = torch.ones(image_embeds_tmp.size()[:-1], dtype=torch.long).to(
                image.device
            )
            image_embeds.append(image_embeds_tmp)
            image_atts.append(image_atts_tmp)
            sam_embeds.append(sam_embeds_tmp)
            sam_atts.append(sam_atts_tmp)
        
        image_embeds = torch.cat(image_embeds, dim=1)
        image_atts = torch.cat(image_atts, dim=1)
        sam_embeds = torch.cat(sam_embeds, dim=1)
        sam_atts = torch.cat(sam_atts, dim=1)
        # print("sam_embeds: ", sam_embeds.shape)
        # print("image_embeds: ", image_embeds.shape)
        # print("sam_atts: ", sam_atts.shape)
        # print("inage_atts: ", image_atts.shape)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # SAM
        sam_query_tokens = self.sam_query_tokens.expand(sam_embeds.shape[0], -1, -1)
        sam_query_output = self.sam_Qformer.bert(
            query_embeds=sam_query_tokens,
            encoder_hidden_states=sam_embeds,
            encoder_attention_mask=sam_atts,
            return_dict=True,
        )
        # print("sam_query_output: ", sam_query_output.last_hidden_state.shape)
        # print("query_output: ", query_output.last_hidden_state.shape)
        # END SAM

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        # SAM
        sam_inputs_opt = self.sam_opt_proj(sam_query_output.last_hidden_state)
        sam_atts_opt = torch.ones(sam_inputs_opt.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # print("sam_inputs_opt: ", sam_inputs_opt.shape)
        # print("inputs_opt: ", inputs_opt.shape)
        # print("sam_atts_opt: ", sam_atts_opt.shape)
        # print("atts_opt: ", atts_opt.shape)
        # END SAM

        self.opt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        # targets = torch.cat([empty_targets, targets], dim=1)
        
        # SAM
        sam_empty_targets = (
            torch.ones(sam_atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        # print("targets:")
        # print(empty_targets.shape)
        # print(sam_empty_targets.shape)
        # print(targets.shape)
        # lwj revised here
        targets = torch.cat([empty_targets, sam_empty_targets, targets], dim=1)
        # targets = torch.cat([ sam_empty_targets, targets], dim=1)
        # lwj revsie here, use sam
        # targets = torch.cat([sam_empty_targets, targets], dim=1)
        # END SAM


        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)

        # inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        # attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        # SAM
        # only sam 

        # weijie revised for only sam
        inputs_embeds = torch.cat([inputs_opt, sam_inputs_opt, inputs_embeds], dim=1)
        # print("showing shape:", inputs_opt.shape, sam_inputs_opt.shape, inputs_embeds.shape)
        # inputs_embeds = torch.cat([sam_inputs_opt, inputs_embeds], dim=1)


        attention_mask = torch.cat([atts_opt, sam_atts_opt, opt_tokens.attention_mask], dim=1)
        # print("showing shape 2:", atts_opt.shape, sam_atts_opt.shape, opt_tokens.attention_mask.shape)
        # attention_mask = torch.cat([sam_atts_opt, opt_tokens.attention_mask], dim=1)

        # print("showing shape:", inputs_embeds.shape, attention_mask.shape)

        # END SAM

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        # print("Forward Finished")
        return {"loss": loss}


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=15,
        max_length=50,
        min_length=10,
        top_p=0.9,
        repetition_penalty=9.0,
        length_penalty=1.2,
        num_captions=1,
        temperature=0.5,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"].view(1, 1, 3, 364, 364)
        img_per_sample = image.shape[1]

        with self.maybe_autocast():
            image_embeds = []
            image_atts = []
            sam_embeds = []
            sam_atts = []
            for i in range(img_per_sample):
                image_embeds_tmp = self.ln_vision(self.visual_encoder(image[:, i, :, :, :]))
                image_atts_tmp = torch.ones(image_embeds_tmp.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                
                # SAM omit layer norm at the output
                to_pil = ToPILImage()
                sam_image = to_pil(image[0, i, :, :, :])
                sam_inputs = self.sam_processor(sam_image, return_tensors="pt").to(image.device)
                sam_image_embeddings = self.sam_model.get_image_embeddings(sam_inputs["pixel_values"]).detach()
                sam_embeddings_shape = sam_image_embeddings.shape

                sam_embeds_tmp = sam_image_embeddings.view(sam_embeddings_shape[0], sam_embeddings_shape[1], -1)
                sam_embeds_tmp = sam_embeds_tmp.to(image.device)
                # print("sam_embeds.shape: ",sam_embeds.shape)
                sam_atts_tmp = torch.ones(sam_embeds_tmp.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                image_embeds.append(image_embeds_tmp)
                image_atts.append(image_atts_tmp)
                sam_embeds.append(sam_embeds_tmp)
                sam_atts.append(sam_atts_tmp)

            # END SAM

            image_embeds = torch.cat(image_embeds, dim=1)
            image_atts = torch.cat(image_atts, dim=1)
            sam_embeds = torch.cat(sam_embeds, dim=1)
            sam_atts = torch.cat(sam_atts, dim=1)
            # print("sam_atts.shape: ",sam_atts.shape)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            # SAM
            sam_query_tokens = self.sam_query_tokens.expand(sam_embeds.shape[0], -1, -1)
            # print("sam_query_tokens.shape: ",sam_query_tokens.shape)
            sam_query_output = self.sam_Qformer.bert(
                query_embeds=sam_query_tokens,
                encoder_hidden_states=sam_embeds,
                encoder_attention_mask=sam_atts,
                return_dict=True,
            )

            # END SAM
            # print("sam_query_output.last_hidden_state.shape: ",sam_query_output.last_hidden_state.shape)

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            # SAM
            sam_inputs_opt = self.sam_opt_proj(sam_query_output.last_hidden_state)
            # print("sam_inputs_opt.shape: ",sam_inputs_opt.shape)
            sam_atts_opt = torch.ones(sam_inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )
            # print("sam_atts_opt.shape: ",sam_atts_opt.shape)
            # END SAM

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            # OLD
            # attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            # END OLD

            # SAM
            # weijie revised for only sam

            attention_mask = torch.cat([atts_opt, sam_atts_opt, opt_tokens.attention_mask], dim=1)
            # attention_mask = torch.cat([sam_atts_opt, opt_tokens.attention_mask], dim=1)
            # END SAM

            
            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)

            # SAM
            inputs_embeds = torch.cat([inputs_opt, sam_inputs_opt, inputs_embeds],dim=1)
            # print("inputs_embeds.shape: ",inputs_embeds.shape)
            # inputs_embeds = torch.cat([sam_inputs_opt, inputs_embeds],dim=1)
            # END SAM

            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
                            
            # previous version for transformers<4.27
            # if use_nucleus_sampling:
            #     query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
            #     num_beams = 1
            # else:
            #     query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            # outputs = self.opt_model.generate(
            #     input_ids=input_ids,
            #     query_embeds=query_embeds,
            #     attention_mask=attention_mask,
            #     do_sample=use_nucleus_sampling,
            #     top_p=top_p,
            #     temperature=temperature,
            #     num_beams=num_beams,
            #     max_new_tokens=max_length,
            #     min_length=min_length,
            #     eos_token_id=self.eos_token_id,
            #     repetition_penalty=repetition_penalty,
            #     length_penalty=length_penalty,
            #     num_return_sequences=num_captions,
            # )

            # prompt_length = opt_tokens.input_ids.shape[1]
            # output_text = self.opt_tokenizer.batch_decode(
            #     outputs[:, prompt_length:], skip_special_tokens=True
            # )
            
            output_text = [text.strip() for text in output_text]
            output_dict = self.output_dict
            self.output_dict = {}
            # return output_text, output_dict
            return output_text
        
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            # SAM
            to_pil = ToPILImage()
            sam_image = to_pil(image[0])
            sam_inputs = self.sam_processor(sam_image, return_tensors="pt").to(image.device)
            sam_image_embeddings = self.sam_model.get_image_embeddings(sam_inputs["pixel_values"]).detach()
            sam_embeddings_shape = sam_image_embeddings.shape

            sam_embeds = sam_image_embeddings.view(sam_embeddings_shape[0], sam_embeddings_shape[1], -1)
            sam_embeds = sam_embeds.to(image.device)
            sam_atts = torch.ones(sam_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            # END SAM

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # SAM
            sam_query_tokens = self.sam_query_tokens.expand(sam_embeds.shape[0], -1, -1)
            sam_query_output = self.sam_Qformer.bert(
                query_embeds=sam_query_tokens,
                encoder_hidden_states=sam_embeds,
                encoder_attention_mask=sam_atts,
                return_dict=True,
            )
            # END SAM

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            # SAM
            sam_inputs_opt = self.sam_opt_proj(sam_query_output.last_hidden_state)
            sam_atts_opt = torch.ones(sam_inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )
            # END SAM

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
        
            # attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            # SAM
            attention_mask = torch.cat([atts_opt, sam_atts_opt, opt_tokens.attention_mask], dim=1)
            # END SAM
            
            # require transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            # inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            # SAM
            inputs_embeds = torch.cat([inputs_opt, sam_inputs_opt, inputs_embeds],dim=1)
            # END SAM
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
        
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 320)
        
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)


        return model
