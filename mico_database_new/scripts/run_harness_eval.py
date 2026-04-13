import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple
from urllib import request, error

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "references" / "harness-evals"
DEFAULT_BASE_URL = "http://localhost:8088"
DEFAULT_REPORT = EVAL_DIR / "latest-report.json"
DEFAULT_BAD_CASES = EVAL_DIR / "latest-bad-cases.json"

SUITES = {
    "patient": EVAL_DIR / "patient-analysis.jsonl",
    "prediction": EVAL_DIR / "prediction-analysis.jsonl",
    "dashboard": EVAL_DIR / "dashboard-analysis.jsonl",
    "harness": EVAL_DIR / "harness-connector.jsonl",
}

CORE_TRACE_STEPS = [
    "context_merged",
    "intent_resolved",
    "task_packet_created",
    "workflow",
    "knowledge_augmentation",
    "hybrid_orchestration",
    "multi_agent_orchestration",
]


def load_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def http_json(url: str, payload: Optional[dict] = None, method: str = "GET") -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url=url, data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_ai(base_url: str, item: dict) -> Tuple[str, dict]:
    session_id = "eval-{0}".format(item["id"])
    payload = {
        "sessionId": session_id,
        "message": item["question"],
        "context": item.get("context") or {},
    }
    response = http_json(base_url.rstrip("/") + "/api/ai/chat", payload=payload, method="POST")
    return session_id, response


def call_mcp_sequence(base_url: str, session_id: str, plan: List[dict]) -> List[dict]:
    results: List[dict] = []
    for index, step in enumerate(plan, start=1):
        step_type = step.get("type", "")
        if step_type == "tool":
            payload = {
                "jsonrpc": "2.0",
                "id": index,
                "method": "tools/call",
                "params": {
                    "name": step["name"],
                    "arguments": merge_session_id(step.get("arguments") or {}, session_id),
                },
            }
        elif step_type == "resource":
            payload = {
                "jsonrpc": "2.0",
                "id": index,
                "method": "resources/read",
                "params": {
                    "uri": step["uri"],
                    "sessionId": session_id,
                },
            }
        elif step_type == "prompt":
            payload = {
                "jsonrpc": "2.0",
                "id": index,
                "method": "prompts/get",
                "params": {
                    "name": step["name"],
                    "arguments": merge_session_id(step.get("arguments") or {}, session_id),
                },
            }
        else:
            raise ValueError("Unsupported mcp plan step type: {0}".format(step_type))

        result = http_json(base_url.rstrip("/") + "/mcp", payload=payload, method="POST")
        results.append({
            "type": step_type,
            "request": payload,
            "response": result,
        })
    return results


def merge_session_id(arguments: dict, session_id: str) -> dict:
    merged = dict(arguments)
    merged.setdefault("sessionId", session_id)
    return merged


def fetch_trace(base_url: str, session_id: str) -> dict:
    return http_json(base_url.rstrip("/") + "/api/ai/traces/{0}/latest".format(session_id))


def flatten_response(resp: dict) -> str:
    parts: List[str] = []
    for key in ("summary", "intent"):
        value = resp.get(key)
        if value:
            parts.append(str(value))
    for card in resp.get("cards") or []:
        if isinstance(card, dict):
            parts.extend(str(card.get(k, "")) for k in ("type", "title", "content"))
    for action in resp.get("actions") or []:
        parts.append(str(action))
    metadata = resp.get("metadata") or {}
    for key in ("analysisLayers", "evidenceChannels", "agentInsights", "review"):
        value = metadata.get(key)
        if value:
            parts.append(json.dumps(value, ensure_ascii=False))
    return "\n".join(parts)


def normalize_step_names(trace: dict) -> List[str]:
    steps = trace.get("steps") or []
    names: List[str] = []
    for step in steps:
        if isinstance(step, dict) and step.get("name"):
            names.append(str(step["name"]))
    return names


def evaluate_response_text(item: dict, resp: dict) -> dict:
    text = flatten_response(resp)
    must_hits = [token for token in item.get("expected_must", []) if token in text]
    should_hits = [token for token in item.get("expected_should", []) if token in text]
    forbidden_hits = [token for token in item.get("forbidden", []) if token in text]
    return {
        "must_hits": must_hits,
        "should_hits": should_hits,
        "forbidden_hits": forbidden_hits,
        "missing_must": [token for token in item.get("expected_must", []) if token not in must_hits],
        "missing_should": [token for token in item.get("expected_should", []) if token not in should_hits],
        "text": text,
    }


def evaluate_trace(item: dict, resp: dict, trace: dict) -> dict:
    metadata = resp.get("metadata") or {}
    trace_steps = normalize_step_names(trace)
    issues: List[str] = []

    if not trace:
        issues.append("missing_trace")
        return {
            "issues": issues,
            "step_names": [],
            "missing_steps": CORE_TRACE_STEPS,
            "review": {},
            "mcp_usage": {},
            "mcp_accesses": [],
        }

    missing_steps = [name for name in CORE_TRACE_STEPS if name not in trace_steps]
    issues.extend("missing_step:{0}".format(name) for name in missing_steps)

    task_packet = metadata.get("taskPacket") or {}
    execution_policy = metadata.get("executionPolicy") or {}
    evidence_policy = metadata.get("evidencePolicy") or {}
    review = metadata.get("review") or trace.get("review") or {}
    mcp_usage = trace.get("mcpUsage") or review.get("mcpUsage") or {}
    mcp_accesses = trace.get("mcpAccesses") or []

    if not task_packet:
        issues.append("missing_task_packet")
    if not execution_policy:
        issues.append("missing_execution_policy")
    if not evidence_policy:
        issues.append("missing_evidence_policy")
    if not review:
        issues.append("missing_review")

    if not metadata.get("harnessRuntime"):
        issues.append("missing_harness_runtime_flag")
    if not metadata.get("hybridMode"):
        issues.append("missing_hybrid_mode")
    if not metadata.get("multiAgentMode"):
        issues.append("missing_multi_agent_mode")

    intent = item.get("intent", "")
    if intent in {"PATIENT_ANALYSIS", "PREDICTION_ASSISTANT"} and not review.get("usedPrediction"):
        issues.append("prediction_layer_not_used")
    if not review.get("usedKnowledge"):
        issues.append("knowledge_layer_not_used")

    evidence_channels = metadata.get("evidenceChannels") or []
    if not evidence_channels:
        issues.append("missing_evidence_channels")

    expected_mcp_usage = item.get("expected_mcp_usage") or {}
    for channel, expected_value in expected_mcp_usage.items():
        actual_value = int(mcp_usage.get(channel, 0) or 0)
        if actual_value < int(expected_value):
            issues.append("missing_mcp_usage:{0}".format(channel))

    expected_mcp_steps = item.get("expected_mcp_steps") or []
    for step_name in expected_mcp_steps:
        if step_name not in trace_steps:
            issues.append("missing_mcp_step:{0}".format(step_name))

    return {
        "issues": issues,
        "step_names": trace_steps,
        "missing_steps": missing_steps,
        "review": review,
        "task_packet": task_packet,
        "mcp_usage": mcp_usage,
        "mcp_accesses": mcp_accesses,
    }


def classify_severity(response_check: dict, trace_check: dict, runtime_error: Optional[str]) -> str:
    if runtime_error:
        return "P0"
    if response_check["forbidden_hits"]:
        return "P0"
    if any(issue in trace_check["issues"] for issue in (
        "missing_trace",
        "missing_task_packet",
        "missing_execution_policy",
        "missing_evidence_policy",
    )):
        return "P0"
    if response_check["missing_must"]:
        return "P1"
    if any(issue.startswith("missing_step:") for issue in trace_check["issues"]):
        return "P1"
    if any(issue.startswith("missing_mcp_usage:") for issue in trace_check["issues"]):
        return "P1"
    if any(issue.startswith("missing_mcp_step:") for issue in trace_check["issues"]):
        return "P1"
    if "prediction_layer_not_used" in trace_check["issues"] or "knowledge_layer_not_used" in trace_check["issues"]:
        return "P1"
    if response_check["missing_should"] or trace_check["issues"]:
        return "P2"
    return "PASS"


def dedupe_preserve(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def suggest_fix_layers(item: dict, response_check: dict, trace_check: dict, severity: str, runtime_error: Optional[str]) -> dict:
    layers: List[str] = []
    reasons: List[str] = []

    if runtime_error or "request_failed" in trace_check.get("issues", []):
        layers.extend(["runtime-ops", "connectivity"])
        reasons.append("请求阶段已失败，先检查服务可用性、端口和接口连通性。")

    runtime_issues = {"missing_trace", "missing_task_packet", "missing_execution_policy", "missing_evidence_policy", "missing_review"}
    if runtime_issues.intersection(trace_check.get("issues", [])):
        layers.append("harness-runtime")
        reasons.append("运行时核心对象缺失，优先检查 Task Packet、Policy、Trace、Review 是否正确挂载。")

    if any(issue.startswith("missing_step:") for issue in trace_check.get("issues", [])):
        layers.append("workflow-orchestration")
        reasons.append("主编排步骤不完整，优先检查 workflow、hybrid、multi-agent 主链路。")

    if "prediction_layer_not_used" in trace_check.get("issues", []):
        layers.extend(["prediction-pipeline", "workflow-orchestration"])
        reasons.append("样本或预测层没有真正命中，优先检查 sample 粒度、预测入口和模型代理链路。")

    if "knowledge_layer_not_used" in trace_check.get("issues", []):
        layers.append("knowledge-rag")
        reasons.append("知识层未命中，优先检查知识库检索、增强拼接和相关 prompt。")

    if any(issue.startswith("missing_mcp_usage:") or issue.startswith("missing_mcp_step:") for issue in trace_check.get("issues", [])):
        layers.append("mcp-connector")
        reasons.append("MCP 命中不足，优先检查 tool/resource/prompt 是否被触发并正确带上 sessionId。")

    if "missing_hybrid_mode" in trace_check.get("issues", []) or "missing_multi_agent_mode" in trace_check.get("issues", []):
        layers.append("agent-orchestration")
        reasons.append("混合编排或多智能体层没有正确进入结果，优先检查总控组装逻辑。")

    if response_check.get("forbidden_hits"):
        layers.extend(["prompt-policy", "review-loop"])
        reasons.append("出现禁用表达，优先收紧系统提示词、边界约束和 review 规则。")

    if response_check.get("missing_must"):
        layers.append("response-synthesis")
        reasons.append("关键回答要点缺失，优先检查总结层是否把数据、知识和预测真正说出来。")

    if response_check.get("missing_should") and not response_check.get("missing_must"):
        layers.append("prompt-tuning")
        reasons.append("回答基础正确但不够完整，可优先优化总结提示词和卡片表达。")

    if not layers:
        layers.append("manual-review")
        reasons.append("未命中明确规则，建议人工检查具体 bad case 文本与 trace。")

    layers = dedupe_preserve(layers)
    primary = layers[0]
    return {
        "primary_layer": primary,
        "recommended_layers": layers,
        "rationale": reasons,
        "next_action": next_action_for_layer(primary, item),
    }


def next_action_for_layer(layer: str, item: dict) -> str:
    return {
        "runtime-ops": "先确认 ai-orchestrator 是否启动，/api/ai/health 是否正常。",
        "connectivity": "先确认主项目、编排服务、模型服务之间的接口联通。",
        "harness-runtime": "先看 Task Packet、executionPolicy、evidencePolicy、review 是否出现在 metadata 与 trace 中。",
        "workflow-orchestration": "先看 workflow / knowledge augmentation / hybrid / multi-agent 这些步骤是否在 trace 中连续出现。",
        "prediction-pipeline": "先检查当前 case 的 patientId/sampleId 是否单一样本，并确认 prediction 链路真的被触发。",
        "knowledge-rag": "先检查知识库命中、knowledgeHits 和知识增强服务的拼接逻辑。",
        "mcp-connector": "先按同一个 sessionId 手动调一次 MCP tool/resource/prompt，确认 trace 里出现对应 mcp_* 步骤。",
        "agent-orchestration": "先检查 HybridResponseService 和 MultiAgentOrchestratorService 的输出是否被覆盖或丢失。",
        "prompt-policy": "先收紧边界提示词，避免把预测写成诊断或把证据层混成事实层。",
        "review-loop": "先检查 review 规则是否正确识别风险、知识命中和模型命中。",
        "response-synthesis": "先看最终 summary/cards/actions 是否真正覆盖 case 的 expected_must。",
        "prompt-tuning": "先优化总结层提示词，让 should 项更稳定命中。",
        "manual-review": "先人工查看这条样例的回答文本和 trace 细节。",
    }.get(layer, "先人工查看这条样例的回答文本和 trace 细节。")


def build_bad_case(item: dict, response_check: dict, trace_check: dict, severity: str, runtime_error: Optional[str]) -> dict:
    suggestion = suggest_fix_layers(item, response_check, trace_check, severity, runtime_error)
    return {
        "id": item["id"],
        "intent": item["intent"],
        "question": item["question"],
        "severity": severity,
        "notes": item.get("notes", ""),
        "runtime_error": runtime_error,
        "missing_must": response_check.get("missing_must", []),
        "forbidden_hits": response_check.get("forbidden_hits", []),
        "trace_issues": trace_check.get("issues", []),
        "missing_trace_steps": trace_check.get("missing_steps", []),
        "mcp_usage": trace_check.get("mcp_usage", {}),
        "review": trace_check.get("review", {}),
        "fix_suggestion": suggestion,
    }


def evaluate_item(item: dict, session_id: str, resp: dict, trace: dict, mcp_results: List[dict]) -> Tuple[dict, Optional[dict]]:
    response_check = evaluate_response_text(item, resp)
    trace_check = evaluate_trace(item, resp, trace)
    severity = classify_severity(response_check, trace_check, None)
    passed = severity == "PASS"
    suggestion = suggest_fix_layers(item, response_check, trace_check, severity, None)

    result = {
        "id": item["id"],
        "sessionId": session_id,
        "intent": item["intent"],
        "passed": passed,
        "severity": severity,
        "must_hit_count": len(response_check["must_hits"]),
        "must_total": len(item.get("expected_must", [])),
        "should_hit_count": len(response_check["should_hits"]),
        "should_total": len(item.get("expected_should", [])),
        "forbidden_hit_count": len(response_check["forbidden_hits"]),
        "forbidden_hits": response_check["forbidden_hits"],
        "must_hits": response_check["must_hits"],
        "should_hits": response_check["should_hits"],
        "missing_must": response_check["missing_must"],
        "missing_should": response_check["missing_should"],
        "trace_issues": trace_check["issues"],
        "trace_steps": trace_check["step_names"],
        "trace_missing_steps": trace_check["missing_steps"],
        "review": trace_check["review"],
        "mcp_usage": trace_check["mcp_usage"],
        "mcp_access_count": len(trace_check["mcp_accesses"]),
        "mcp_plan_count": len(mcp_results),
        "fix_suggestion": suggestion,
        "notes": item.get("notes", ""),
    }

    if not passed:
        return result, build_bad_case(item, response_check, trace_check, severity, None)
    return result, None


def evaluate_runtime_error(item: dict, err: str) -> Tuple[dict, dict]:
    response_check = {
        "missing_must": item.get("expected_must", []),
        "forbidden_hits": [],
    }
    trace_check = {
        "issues": ["request_failed"],
        "missing_steps": CORE_TRACE_STEPS,
        "review": {},
        "mcp_usage": {},
    }
    severity = classify_severity(response_check, trace_check, err)
    suggestion = suggest_fix_layers(item, response_check, trace_check, severity, err)
    result = {
        "id": item["id"],
        "sessionId": "eval-{0}".format(item["id"]),
        "intent": item["intent"],
        "passed": False,
        "severity": severity,
        "error": err,
        "must_hit_count": 0,
        "must_total": len(item.get("expected_must", [])),
        "should_hit_count": 0,
        "should_total": len(item.get("expected_should", [])),
        "forbidden_hit_count": 0,
        "forbidden_hits": [],
        "must_hits": [],
        "should_hits": [],
        "missing_must": item.get("expected_must", []),
        "missing_should": item.get("expected_should", []),
        "trace_issues": ["request_failed"],
        "trace_steps": [],
        "trace_missing_steps": CORE_TRACE_STEPS,
        "review": {},
        "mcp_usage": {},
        "mcp_access_count": 0,
        "mcp_plan_count": 0,
        "fix_suggestion": suggestion,
        "notes": item.get("notes", ""),
    }
    bad_case = build_bad_case(item, response_check, trace_check, severity, err)
    return result, bad_case


def run_suite(base_url: str, path: Path) -> Tuple[List[dict], List[dict]]:
    results: List[dict] = []
    bad_cases: List[dict] = []
    for item in load_jsonl(path):
        try:
            session_id, resp = call_ai(base_url, item)
            mcp_results = call_mcp_sequence(base_url, session_id, item.get("mcp_plan") or [])
            trace = fetch_trace(base_url, session_id)
            result, bad_case = evaluate_item(item, session_id, resp, trace, mcp_results)
            results.append(result)
            if bad_case is not None:
                bad_cases.append(bad_case)
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            result, bad_case = evaluate_runtime_error(item, str(exc))
            results.append(result)
            bad_cases.append(bad_case)
    return results, bad_cases


def summarize_suite(results: List[dict]) -> dict:
    severity_counts = Counter(result["severity"] for result in results)
    layer_counts = Counter(result.get("fix_suggestion", {}).get("primary_layer", "") for result in results if not result.get("passed"))
    return {
        "total": len(results),
        "passed": sum(1 for result in results if result.get("passed")),
        "failed": sum(1 for result in results if not result.get("passed")),
        "severity": dict(severity_counts),
        "primaryFixLayers": dict(layer_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run harness eval suites against ai-orchestrator")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--suite", default="all", choices=["all", "patient", "prediction", "dashboard", "harness"])
    parser.add_argument("--out", default=str(DEFAULT_REPORT))
    parser.add_argument("--bad-cases-out", default=str(DEFAULT_BAD_CASES))
    args = parser.parse_args()

    suite_names = list(SUITES.keys()) if args.suite == "all" else [args.suite]
    report = {
        "baseUrl": args.base_url,
        "suites": {},
        "summary": {"total": 0, "passed": 0, "failed": 0, "severity": {}, "primaryFixLayers": {}},
    }
    all_bad_cases: List[dict] = []
    global_severity: Counter = Counter()
    global_layers: Counter = Counter()

    for name in suite_names:
        results, bad_cases = run_suite(args.base_url, SUITES[name])
        suite_summary = summarize_suite(results)
        report["suites"][name] = {
            "path": str(SUITES[name]),
            "summary": suite_summary,
            "results": results,
        }
        report["summary"]["total"] += suite_summary["total"]
        report["summary"]["passed"] += suite_summary["passed"]
        report["summary"]["failed"] += suite_summary["failed"]
        global_severity.update(suite_summary["severity"])
        global_layers.update(suite_summary["primaryFixLayers"])
        for case in bad_cases:
            case["suite"] = name
            all_bad_cases.append(case)

    report["summary"]["severity"] = dict(global_severity)
    report["summary"]["primaryFixLayers"] = dict(global_layers)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    bad_case_report = {
        "baseUrl": args.base_url,
        "totalBadCases": len(all_bad_cases),
        "severity": dict(Counter(case["severity"] for case in all_bad_cases)),
        "primaryFixLayers": dict(Counter(case["fix_suggestion"]["primary_layer"] for case in all_bad_cases)),
        "badCases": all_bad_cases,
    }
    bad_case_path = Path(args.bad_cases_out)
    bad_case_path.parent.mkdir(parents=True, exist_ok=True)
    bad_case_path.write_text(json.dumps(bad_case_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Eval completed: {0}/{1} passed".format(report["summary"]["passed"], report["summary"]["total"]))
    for suite_name, suite in report["suites"].items():
        summary = suite["summary"]
        print("- {0}: {1}/{2} passed, severity={3}, fixLayers={4}".format(
            suite_name,
            summary["passed"],
            summary["total"],
            summary["severity"],
            summary["primaryFixLayers"],
        ))
    print("Report: {0}".format(out_path))
    print("Bad cases: {0}".format(bad_case_path))

    if report["summary"]["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()