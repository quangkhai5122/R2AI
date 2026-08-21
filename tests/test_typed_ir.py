import pytest

from vifinqa.codegen.typed_ir import FactBinding, IRValidationError, compile_program, program_for_operation


def _fact(name, value, typ="money", var="df1", col=1, row=1, year=2024):
    return FactBinding(name=name, expr=f"float({var}['value'].iloc[0]) * 1e9", value=value,
                       scalar_type=typ, stable_cell=(f"AAA_{year}", row, name, col), confidence=95.0)


def test_difference_is_typed_and_unit_normalized():
    facts=[_fact("a",120e9),_fact("b",100e9,var="df2",row=2)]
    c=compile_program(program_for_operation("difference",["a","b"],{"output_type":"number"}),facts,{"output_type":"number","unit_scale":1e9})
    assert c.answer == 20.0
    assert c.inferred_type == "money"
    assert c.referenced_facts == ("a","b")


def test_growth_and_ratio_are_typed():
    facts=[_fact("end",120e9),_fact("base",100e9,var="df2",row=2)]
    growth=compile_program(program_for_operation("growth_pct",["end","base"],{"output_type":"percent"}),facts,{"output_type":"percent","unit_scale":1},"tang truong")
    assert growth.answer == 20.0
    ratio=compile_program(program_for_operation("ratio",["end","base"],{"output_type":"ratio"}),facts,{"output_type":"ratio","unit_scale":1},"he so")
    assert ratio.answer == 1.2


def test_argmax_projects_result_not_score():
    facts=[_fact("score_a",2e9,row=1),_fact("result_a",90e9,row=2),_fact("score_b",3e9,var="df2",row=3),_fact("result_b",70e9,var="df2",row=4)]
    program={"schema_version":1,"output_type":"number","root":{"op":"argmax_project","items":[{"score":{"op":"ref","fact":"score_a"},"result":{"op":"ref","fact":"result_a"}},{"score":{"op":"ref","fact":"score_b"},"result":{"op":"ref","fact":"result_b"}}]}}
    c=compile_program(program,facts,{"output_type":"number","unit_scale":1e9})
    assert c.answer == 70.0
    assert c.root_op == "argmax_project"


def test_duplicate_stable_cells_fail_closed():
    facts=[_fact("a",1e9),_fact("b",2e9)]
    facts=[FactBinding(f.name,f.expr,f.value,f.scalar_type,("same",1,"same",1),f.confidence) for f in facts]
    with pytest.raises(IRValidationError,match="duplicate stable"):
        compile_program(program_for_operation("difference",["a","b"],{"output_type":"number"}),facts,{"output_type":"number","unit_scale":1e9})


def test_unused_fact_fails_closed():
    facts=[_fact("a",1e9),_fact("b",2e9),_fact("unused",3e9,row=3)]
    program=program_for_operation("difference",["a","b"],{"output_type":"number"})
    program["root"]={"op":"difference","args":[{"op":"ref","fact":"a"},{"op":"ref","fact":"b"}]}
    with pytest.raises(IRValidationError,match="unused facts"):
        compile_program(program,facts,{"output_type":"number","unit_scale":1e9})


def test_ratio_unit_mismatch_fails_closed():
    facts=[_fact("a",1e9),_fact("b",2e9,var="df2",row=2)]
    with pytest.raises(IRValidationError,match="declared percent != route ratio"):
        compile_program(program_for_operation("ratio",["a","b"],{"output_type":"percent"}),facts,{"output_type":"ratio","unit_scale":1})
