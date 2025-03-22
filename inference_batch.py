import argparse
import json
import random
import copy
import time
import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from pipeline.common.config import Config
from pipeline.common.dist_utils import get_rank
from pipeline.common.registry import registry
from pipeline.conversation.conversation_batch import Chat, CONV_VISION
from pipeline.datasets.datasets.multimodal_dataset import MultimodalDataset 
from torch.utils.data import DataLoader

# imports modules for registration
from pipeline.datasets.builders import *
from pipeline.models import *
from pipeline.processors import *
from pipeline.runners import *
from pipeline.tasks import *

PROMPT_TEMPLATE = ("You are provided with two drugs: <compound1><compoundHere></compound1> "
                   "and <compound2><compoundHere></compound2>. Analyze the given compounds and predict "
                   "the drug interactions between them.")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num_beams", type=int, default=1, help="specify the num_beams for text generation.")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1, help="specify the temperature for text generation.")
    parser.add_argument("--out_file", type=str, default="xxx.json", help="specify the output file.")
    parser.add_argument("--in_file_folder", type=str, default="aaa.json", help="specify the input file or dataset path.")
    parser.add_argument("--batch_size", type=int, default=4, help="specify the batch size for inference.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)
use_amp = cfg.run_cfg.get("amp", False)
amp_encoder = cfg.run_cfg.get("amp_encoder", use_amp)
amp_proj = cfg.run_cfg.get("amp_proj", use_amp)

in_file_folder = args.in_file_folder
batch_size = args.batch_size
model_config = cfg.model_cfg
image_size = model_config.get('image_size', 224)
use_image = True if "image_mol" in model_config.get('encoder_names', "") else False
use_graph = True if "gnn" in model_config.get('encoder_names', "") else False
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
print(model_config)
model = model_cls.from_config(model_config)
if model.lora_rank:
    print("merge_and_unload LoRA")
    model.llama_model = model.llama_model.merge_and_unload()

model = model.to('cuda:{}'.format(args.gpu_id)).eval()

chat = Chat(model, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Batch Inference Function
# ========================================

def infer_QA():
    with open("prompts/alignment.txt", "r") as f:
        prompt = f.read()
    # Initialize the dataset
    dataset = MultimodalDataset(
        datapath=in_file_folder,
        use_image=use_image,
        use_graph=use_graph,
        image_size=image_size,
        is_train=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MultimodalDataset.collater
    )
    
    out = {}
    sample_counter = 0
    # Iterate over batches with a tqdm progress bar
    with torch.no_grad(): 
        for batch in tqdm.tqdm(dataloader, desc="Running inference", unit="batch"):
            # Expecting batch to be a dict with keys "text_input" and "image"
            batch_smiles = batch.get("smiles", None)  # list of SMILES strings, one per sample in the batch
            # We assume that batch["image"] is a list of image tensors for each sample.
            # (Each element is typically a tensor of shape (2, C, H, W) if two images are provided.)
            batch_imgs = batch.get("image", None)
            batch_graphs = batch.get("graph", None)
            ground_truth_texts = batch.get("text_input", None)  # list of questions, one per sample in the batch
            # Create a conversation for each sample and compute the image embedding.
            convs = []
            img_embeddings = []
            for i in range(len(batch_smiles)):
                # Create a fresh conversation instance from the template.
                conv = CONV_VISION.copy()
                # Append the initial prompt message (as in your upload_img routine).
                conv.append_message(conv.roles[0], 
                                    prompt)
                # Prepare the input dictionary for image encoder.
                inputs = {}
                if batch_imgs is not None:
                    # Here, batch_imgs[i] is the transformed image tensor (for the sample).
                    inputs["image"] = batch_imgs[i].to('cuda:{}'.format(args.gpu_id))
                # If graph information exists, you can add it similarly:
                if batch_graphs is not None:
                    inputs["graph"] = batch["graph"][i].to('cuda:{}'.format(args.gpu_id))
                
                if "image" in inputs:
                    # Compute image embedding in a batched manner.
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        # Note: encode_img_infer should accept a dictionary of inputs
                        # and return an embedding tensor.
                        img_emb, _ = chat.model.encode_img_infer(inputs, 
                                                                device=chat.device, 
                                                                autocast=amp_encoder, 
                                                                autocast_proj=amp_proj)
                    img_embeddings.append(img_emb)
                else:
                    # If no image exists, append a placeholder or skip.
                    img_embeddings.append(None)
                convs.append(conv)
            
            # Use the batch version of answer generation.
            # This method uses a list of Conversation objects and a list of image embeddings.
            results, _ = chat.answer_batch(
                convs, 
                img_embeddings,
                num_beams=args.num_beams,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                max_length=2000
            )
            
            # Store each sample's result into the output dictionary.
            for i, generated_text in enumerate(results):
                smiles = batch_smiles[i]
                ground_truth_text = ground_truth_texts[i]
                out[smiles] = [
                    prompt, ground_truth_text, generated_text
                ]
                sample_counter += 1

    with open(args.out_file, "wt") as f:
        json.dump(out, f, indent=4)
    print(f"Inference completed. Results saved to {args.out_file}")

infer_QA()
