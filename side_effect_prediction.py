import pandas as pd
from DDI_model import DDI_model

drug_info = pd.read_csv('./data/twosides_drug_info.csv', index_col=0)
drug_info_dict = dict(zip(drug_info.pubchemID, drug_info.name))

side_effect_info = pd.read_csv('./data/twosides_side_effect_info.csv', index_col=0)
side_effect_dict = dict(zip(side_effect_info.SE_map, side_effect_info['Side Effect Name']))
se_UMLS_id = dict(zip(side_effect_info.SE_id, side_effect_info.SE_map))

ts_exp = pd.read_csv('./data/twosides_predicted_expression_scaled.csv')

ddi_model = DDI_model()
ddi_model.load_model(model_load_path='./trained_model/', model_name='ddi_model_weights.h5',
                     threshold_name='ddi_model_threshold.csv')


def predict(drug1_cid: int, drug2_cid: int, side_effect_UMLS_CUI: str):
    try:
        side_effect_type = se_UMLS_id[side_effect_UMLS_CUI]
    except NameError:
        return {"error": "Side effect type doesn't exist"}

    temp_df = pd.DataFrame({'drug1': drug1_cid, 'drug2': drug2_cid, 'SE': side_effect_type}, index=[0])

    ts_drug_list = ts_exp.pubchem.values.tolist()
    if (temp_df.drug1.values not in ts_drug_list) | (temp_df.drug2.values not in ts_drug_list):
        return {"error": "Drug cannot be found"}

    predicted_label = ddi_model.predict(temp_df, exp_df=ts_exp)

    return {
        "drug_1": drug_info_dict[drug1_cid],
        "drug_2": drug_info_dict[drug2_cid],
        "se": side_effect_dict[side_effect_type],
        "predicted_label": predicted_label.predicted_label[0],
        "predicted_score": predicted_label.predicted_score[0]
    }
