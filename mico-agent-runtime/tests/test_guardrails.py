from mico_agent_runtime.governance.guardrails import evaluate_input


def test_explicit_research_boundaries_do_not_trigger_a_false_rejection() -> None:
    assert evaluate_input(
        "按国家分层比较两个研究组的微生态摘要，不把地区差异解释为因果关系。"
    ).verdict == "ALLOW"
    assert evaluate_input(
        "比较两个研究组的差异，只输出相关性和局限性，不输出因果或诊断结论。"
    ).verdict == "ALLOW"


def test_positive_clinical_or_causal_requests_remain_rejected() -> None:
    assert evaluate_input("根据某患者数据判断他是否患病。").verdict == "REJECT"
    assert evaluate_input("宣布某 species 对疾病具有确定因果作用。").verdict == "REJECT"
    assert evaluate_input(
        "不要作因果解释，但请给出某 species 对疾病的确定因果结论。"
    ).verdict == "REJECT"
