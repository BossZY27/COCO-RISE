import json
import pytest
from scripts.compare_proposal_answers import load_cases


def test_evaluation_requires_human_review_and_explicit_gold_fields(tmp_path):
    path=tmp_path/'cases.json'
    case={'question':'ทุนอะไร','reviewed_by':'Reviewer','expected_status':'insufficient_evidence','expected_numbers':[],'evidence':[]}
    path.write_text(json.dumps([case]))
    assert load_cases(path)==[case]
    path.write_text(json.dumps([{**case,'reviewed_by':''}]))
    with pytest.raises(ValueError,match='reviewed_by'):load_cases(path)
