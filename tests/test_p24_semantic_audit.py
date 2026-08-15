from vifinqa.devset.p24_semantic_audit import COMPLEX_IDS


def test_complex_id_contract_is_stable():
    assert len(COMPLEX_IDS) == 21
    assert {375, 397, 417, 425, 447, 508, 516, 570} <= COMPLEX_IDS
