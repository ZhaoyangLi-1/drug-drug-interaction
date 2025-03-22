from rdkit import Chem
import numpy as np
import json
import pickle
import argparse
import os
from rdkit.Chem.rdchem import BondType, BondDir, ChiralType
from tqdm import tqdm


BOND_TYPE = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3, BondType.DATIVE: 4}
BOND_DIR = {BondDir.NONE: 0, BondDir.ENDUPRIGHT: 1, BondDir.ENDDOWNRIGHT: 2}
CHI = {ChiralType.CHI_UNSPECIFIED: 0, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_TETRAHEDRAL_CCW: 2, ChiralType.CHI_OTHER: 3}

def bond_dir(bond):
    d = bond.GetBondDir()
    return BOND_DIR[d]

def bond_type(bond):
    t = bond.GetBondType()
    return BOND_TYPE[t]

def atom_chiral(atom):
    c = atom.GetChiralTag()
    return CHI[c]

def atom_to_feature(atom):
    num = atom.GetAtomicNum() - 1
    if num == -1:
        # atom.GetAtomicNum() is 0, which is the generic wildcard atom *, may be used to symbolize an unknown atom of any element.
        # See https://biocyc.org/help.html?object=smiles
        num = 118  # normal num is [0, 117], so we use 118 to denote wildcard atom *
    return [num, atom_chiral(atom)]

def bond_to_feature(bond):
    return [bond_type(bond), bond_dir(bond)]

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 2
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph 


def convert_chembl():
    # with open("/home/youwei/project/drugchat/data/ChEMBL_QA_train.json", "rt") as f:
    with open("/home/youwei/project/drugchat/data/PubChem_QA.json", "rt") as f:
        js = json.load(f)
    out = []
    for smi, rec in js.items():
        graph = smiles2graph(smi)
        for question, answer in rec:
            out.append({"graph": graph, "question": question, "answer": str(answer)})

    
    with open("./dataset/PubChem_QA_train.pkl", "wb") as f:
        pickle.dump(out, f)


def is_int(x):
    try:
        x = int(x)
    except:
        return False
    return True


def convert_simple_graph_smi(infile, outfile):
    """
    Convert to a data format that supports both graphs and images (which is converted to feature dataset later).
    Handles cases where the SMILES string contains two SMILES separated by '|'.
    """
    with open(infile, "rt") as f:
        js = json.load(f)
    out = {}
    for smi, rec in tqdm(js.items(), desc="Processing"):
        if is_int(smi):
            smi, rec = rec

        try:
            # Check if there are two SMILES strings separated by '|'
            if "|" in smi:
                smi1, smi2 = smi.split("|")
                try:
                    # Convert both SMILES strings to graphs
                    graph1 = smiles2graph(smi1)
                except Exception as e:
                    raise ValueError(f"Error processing graph1 from SMILES '{smi1}': {str(e)}")

                try:
                    graph2 = smiles2graph(smi2)
                except Exception as e:
                    raise ValueError(f"Error processing graph2 from SMILES '{smi2}': {str(e)}")
                
                out[smi] = {"graph1": graph1, "graph2": graph2, "QA": rec}
            else:
                # Single SMILES string case
                graph = smiles2graph(smi)
                out[smi] = {"graph": graph, "QA": rec}
        except Exception as e:
            print(f"Error processing SMILES '{smi}': {str(e)}")
            continue

    print(f"Total valid entries: {len(out)}")

    with open(outfile, "wb") as f:
        pickle.dump(out, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Converts SMILES dataset to graphs")
    parser.add_argument("--smiles_path", default="data/ChEMBL_QA_test.json", type=str, help="path to json file.")
    parser.add_argument("--save_dir", type=str, help="path to save output.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    outfile = os.path.join(args.save_dir, 'graph_smi.pkl')
    convert_simple_graph_smi(args.smiles_path, outfile)
