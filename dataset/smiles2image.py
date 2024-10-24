import argparse
import json
import os
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm


def Smiles2Img(smis, size=224, savePath=None):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png
    '''
    # try:
    mol = Chem.MolFromSmiles(smis)
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
    if savePath is not None:
        img.save(savePath)
    return img
    # except:
    #     return None


def main(smiles_path, save_dir, start_index):
    with open(smiles_path, "rt") as f:
        js = json.load(f)
    
    os.makedirs(save_dir, exist_ok=True)
    
    out_js = {}
    i = 0
    for _, (smis_string, qa) in enumerate(tqdm(js.items(), desc="Processing")):
        pair_idx = i + start_index
        smis = smis_string.split("|")
        
        for smi_idx, smi in enumerate(smis):
            save_path = os.path.join(save_dir, f"img_{pair_idx}_{smi_idx}.png")
            try:
                Smiles2Img(smi, savePath=save_path)
            except:
                print(f"smiles: {smi}")
                continue
        out_js[pair_idx] = [smis_string, qa]
        i += 1

    out_js_path = os.path.join(save_dir, "smiles_img_qa.json")
    with open(out_js_path, "wt") as f:
        json.dump(out_js, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Converts SMILES dataset to images")
    parser.add_argument("--smiles_path", default="data/ChEMBL_QA_test.json", type=str, help="path to json file.")
    parser.add_argument("--save_dir", default="data/ChEMBL_QA_test_image/", type=str, help="path to save output.")
    parser.add_argument("--start_index", type=int, default=0, help="specify the gpu to load the model.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args.smiles_path, args.save_dir, args.start_index)
