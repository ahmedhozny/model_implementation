import json

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sentence_transformers import SentenceTransformer

from synergy_model import MultiViewNet

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentence_transformer = SentenceTransformer('simcsesqrt-model', device=device)


model = MultiViewNet()
checkpoint = torch.load('mainsplit-attention-comb', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

context_dict: dict = json.loads(open("./drugcombdb/context_set_m.json", 'r').read())
context_fe = [[value] for value in context_dict.values()]

def get_fingerprint(mol):
    return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024)


def smiles_encode(smile_str):
    smile_encoded = sentence_transformer.encode(smile_str)
    mol = Chem.MolFromSmiles(smile_str)
    fp = np.asarray(get_fingerprint(mol))
    return {"drug_smiles": smile_encoded, "drug_fp": fp}


async def predict(smile_1_vectors, smile_2_vectors, fp1_vectors, fp2_vectors):
    context = torch.FloatTensor(context_fe).to(device)
    smile_1_vectors = torch.FloatTensor([smile_1_vectors] * len(context)).to(device)
    smile_2_vectors = torch.FloatTensor([smile_2_vectors] * len(context)).to(device)
    fp1_vectors = torch.FloatTensor([fp1_vectors] * len(context)).to(device)
    fp2_vectors = torch.FloatTensor([fp2_vectors] * len(context)).to(device)

    predicted = model.forward(smile_1_vectors, smile_2_vectors, context, fp1_vectors, fp2_vectors)
    return np.linalg.norm(predicted.cpu().detach().numpy(), "fro")
