from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from synergy_prediction import predict as predict_senergy, smiles_encode
from side_effect_prediction import predict as predict_side_effect


class SideEffectsPrediction(BaseModel):
    drug1_id: int
    drug2_id: int
    side_effect_id: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Model Instance is running"}


@app.post("/synergy")
async def ddi_predict(drug1_smiles: str = Form(...), drug2_smiles: str = Form(...)):
    drug_1 = smiles_encode(drug1_smiles)
    drug_2 = smiles_encode(drug2_smiles)

    res = await predict_senergy(drug_1["drug_smiles"], drug_2["drug_smiles"], drug_1["drug_fp"], drug_2["drug_fp"])
    return {"synergy_score": res}


@app.post("/side_effects")
async def ddi_predict(response: SideEffectsPrediction):

    res = await predict_side_effect(response.drug1_id, response.drug2_id, response.side_effect_id)
    return res
