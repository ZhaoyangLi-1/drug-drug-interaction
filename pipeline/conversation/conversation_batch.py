import os
import pickle
import time
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from torchvision import transforms

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from pipeline.common.registry import registry
from torch_geometric.data import Data, Batch


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


CONV_VISION = Conversation(
    system="",
    roles=["Human", "Assistant"],
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


class Chat:
    def __init__(self, model, vis_processor=None, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' may be encoded in two ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self.transforms = transforms.Compose([
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            normalize,
        ])

    def ask(self, text, conv):
        # If the last human message ended with an image placeholder, append text to that message.
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] and conv.messages[-1][1][-12:] == '</compound2>':
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        """
        Single-sample answer (unchanged). For batch processing, use answer_batch().
        """
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The conversation length exceeds the max length. Some context may be truncated.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]
        if self.model.llama_tokenizer.pad_token is None:
            self.model.llama_tokenizer.pad_token = self.model.llama_tokenizer.eos_token

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            attention_mask=torch.ones(embs.shape[:-1], dtype=torch.long, device=self.device),
            pad_token_id=self.model.llama_tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=False,
            min_new_tokens=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # Remove potential starting <unk> token.
            output_token = output_token[1:]
        if output_token[0] == 1:  # Remove start token <s> if present.
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # Remove the stop sign.
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def answer_batch(self, convs: List[Conversation], img_lists: List[Any],
                     max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                     repetition_penalty=1.0, length_penalty=1, temperature=1.0,
                     max_length=2000):
        """
        Process a batch of conversations in parallel.
        Args:
            convs: List of Conversation objects.
            img_lists: List of image embedding tensors (or lists) for each conversation.
        Returns:
            A tuple (results, outputs) where results is a list of generated texts,
            and outputs is the raw output tokens.
        """
        # Append a placeholder for the assistant's response for each conversation.
        for conv in convs:
            conv.append_message(conv.roles[1], None)
        # Get batched context embeddings.
        embs = self.get_context_emb_batch(convs, img_lists)  # Shape: (batch, seq_len, embed_dim)
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The conversation lengths exceed the max length. Some context may be truncated.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]
        attention_mask = torch.ones(embs.shape[:2], dtype=torch.long, device=self.device)
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            attention_mask=attention_mask,
            pad_token_id=self.model.llama_tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=False,
            min_new_tokens=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        # Process each generated output.
        results = []
        for i, out_tokens in enumerate(outputs):
            if out_tokens[0] == 0:
                out_tokens = out_tokens[1:]
            if out_tokens[0] == 1:
                out_tokens = out_tokens[1:]
            output_text = self.model.llama_tokenizer.decode(out_tokens, add_special_tokens=False)
            output_text = output_text.split('###')[0]
            output_text = output_text.split('Assistant:')[-1].strip()
            convs[i].messages[-1][1] = output_text
            results.append(output_text)
        return results, outputs.cpu().numpy()

    def get_context_emb(self, conv: Conversation, img_list: Any):
        """
        Process a single conversation.
        img_list is expected to be a tensor or list containing the image embeddings.
        """
        # For single sample, assume img_list is a tensor with shape (N, C, H, W) or similar.
        if isinstance(img_list, torch.Tensor):
            split_imgs = [img for img in torch.split(img_list, 1, dim=0)]
        else:
            split_imgs = img_list
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<compoundHere>')
        assert len(prompt_segs) == len(split_imgs) + 1, "Unmatched numbers of image placeholders and images."
        seg_embs = []
        for i, seg in enumerate(prompt_segs):
            tokens = self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=(i == 0)
            ).to(self.device).input_ids
            emb = self.model.llama_model.model.embed_tokens(tokens)
            seg_embs.append(emb)
        # Interleave text embeddings and image embeddings.
        mixed_embs = []
        for i in range(len(split_imgs)):
            mixed_embs.append(seg_embs[i])
            # Ensure the image embedding has a time dimension (unsqueeze if needed).
            img_emb = split_imgs[i]
            if len(img_emb.shape) == 2:
                img_emb = img_emb.unsqueeze(1)
            mixed_embs.append(img_emb)
        mixed_embs.append(seg_embs[-1])
        conv_emb = torch.cat(mixed_embs, dim=1)
        return conv_emb

    def get_context_emb_batch(self, convs: List[Conversation], img_lists: List[Any]):
        """
        Process a batch of conversations.
        For each conversation, compute the context embedding by interleaving the tokenized prompt and image embeddings.
        Then, pad all embeddings to the same sequence length and return a batch tensor.
        """
        all_embs = []
        for conv, img_list in zip(convs, img_lists):
            if isinstance(img_list, torch.Tensor):
                split_imgs = [img for img in torch.split(img_list, 1, dim=0)]
            else:
                split_imgs = img_list
            prompt = conv.get_prompt()
            prompt_segs = prompt.split('<compoundHere>')
            assert len(prompt_segs) == len(split_imgs) + 1, "Unmatched numbers of image placeholders and images."
            seg_embs = []
            for i, seg in enumerate(prompt_segs):
                tokens = self.model.llama_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=(i == 0)
                ).to(self.device).input_ids
                emb = self.model.llama_model.model.embed_tokens(tokens)
                seg_embs.append(emb)
            mixed_embs = []
            for i in range(len(split_imgs)):
                mixed_embs.append(seg_embs[i])
                img_emb = split_imgs[i]
                if len(img_emb.shape) == 2:
                    img_emb = img_emb.unsqueeze(1)
                mixed_embs.append(img_emb)
            mixed_embs.append(seg_embs[-1])
            conv_emb = torch.cat(mixed_embs, dim=1)  # (1, seq_length, embed_dim)
            all_embs.append(conv_emb)
        # Pad all conversation embeddings to the same length.
        max_len = max(emb.shape[1] for emb in all_embs)
        padded_embs = []
        for emb in all_embs:
            pad_len = max_len - emb.shape[1]
            if pad_len > 0:
                pad = torch.zeros(emb.shape[0], pad_len, emb.shape[2], device=emb.device)
                padded_emb = torch.cat([emb, pad], dim=1)
            else:
                padded_emb = emb
            padded_embs.append(padded_emb)
        # Stack into a batch tensor of shape (batch_size, max_len, embed_dim)
        batch_embs = torch.cat(padded_embs, dim=0)
        return batch_embs

    def upload_img(self, image, conv, img_list, autocast=False, autocast_proj=False):
        """
        This function processes a single image input (as a string) by writing a temporary file and waiting
        for an external process to generate a pickle file containing the processed image (and possibly graph) data.
        For batch processing, you can call this method for each sample and collect the results.
        """
        assert isinstance(image, str), f"Expected a string but got {image}"
        timestamp = time.time()
        with open("dataset/tmp_smiles.txt", "wt") as f:
            f.write(str(timestamp) + " " + image)
        inputs = {}
        for _ in range(60):
            time.sleep(1)
            pkl_file = "dataset/tmp_smiles.pkl"
            if os.path.isfile(pkl_file) and os.path.getsize(pkl_file) > 1:
                cnt = 10
                while cnt > 0:
                    try:
                        with open(pkl_file, "rb") as f:
                            res = pickle.load(f)
                        break
                    except EOFError:
                        cnt -= 1
                        continue
                res0 = res[0]
                res1 = res[1]
                t2, t3 = res0["timestamp"], res1["timestamp"]
                if t2 > timestamp and t3 > timestamp:
                    if "graph" in res0 and "graph" in res1:
                        from torch.nn.functional import normalize
                        g1, g2 = res0["graph"], res1["graph"]
                        graph1 = Data(x=torch.asarray(g1['node_feat']),
                                      edge_index=torch.asarray(g1['edge_index']),
                                      edge_attr=torch.asarray(g1['edge_feat']))
                        graph2 = Data(x=torch.asarray(g2['node_feat']),
                                      edge_index=torch.asarray(g2['edge_index']),
                                      edge_attr=torch.asarray(g2['edge_feat']))
                        inputs["graph"] = [graph1, graph2]
                    if "img_save_path" in res0 and "img_save_path" in res1:
                        img0 = Image.open(res0["img_save_path"]).convert("RGB")
                        img1 = Image.open(res1["img_save_path"]).convert("RGB")
                        img0_transformed = self.transforms(img0).unsqueeze(0).to(self.device)
                        img1_transformed = self.transforms(img1).unsqueeze(0).to(self.device)
                        inputs["image"] = torch.cat([img0_transformed, img1_transformed], dim=0)
                    break
        if "image" not in inputs and "graph" not in inputs:
            return  # issues in creating inputs
        image_emb, _ = self.model.encode_img_infer(inputs, device=self.device,
                                                   autocast=autocast, autocast_proj=autocast_proj)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "You are provided with two drugs: <compound1><compoundHere></compound1> <compound2><compoundHere></compound2>. ")
        msg = "Received."
        return msg
