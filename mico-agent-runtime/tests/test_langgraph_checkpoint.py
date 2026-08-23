from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from mico_agent_runtime.contracts.intent import IntentTaskRequest
from mico_agent_runtime.runtime.persistence import RuntimePersistenceCoordinator

from mico_agent_runtime.runtime.checkpoint import EncryptedLangGraphCheckpointSaver
from mico_agent_runtime.runtime.persistence import RuntimePersistenceCoordinator
from mico_agent_runtime.storage.crypto import RuntimeStateCipher, StateBindingContext
from mico_agent_runtime.storage.memory_store import InMemoryRuntimeStore
from mico_agent_runtime.storage.models import AgentRunRecord, RuntimeStatus


RUN = "run-" + "1" * 32
TASK = "task-" + "2" * 32
TRACE = "trace-" + "3" * 32
NOW = datetime(2026, 8, 22, tzinfo=timezone.utc)


class State(TypedDict, total=False):
    value: int


async def prepared_store() -> tuple[InMemoryRuntimeStore, RuntimeStateCipher]:
    store = InMemoryRuntimeStore()
    cipher = RuntimeStateCipher(b"k" * 32, "runtime-key-1")
    await store.create_run(AgentRunRecord(
        dataContractVersion="v1",
        runId=RUN,
        taskId=TASK,
        traceId=TRACE,
        status=RuntimeStatus.RUNNING,
        createdAt=NOW,
        updatedAt=NOW,
        encryptedStatePayload=cipher.encrypt(
            {"request": {"runId": RUN, "taskId": TASK, "traceId": TRACE}},
            StateBindingContext(runId=RUN, taskId=TASK, traceId=TRACE, dataContractVersion="v1"),
        ),
        encryptionKeyId="runtime-key-1",
    ))
    return store, cipher


def intent_request() -> IntentTaskRequest:
    return IntentTaskRequest.model_validate({
        "runId": RUN,
        "taskId": TASK,
        "requesterId": "principal-" + "5" * 32,
            "requestedScopes": ["mico:query:read"],
        "question": "show sample evidence",
            "allowedWorkflows": ["dynamic_read_query"],
        "createdAt": NOW,
        "traceId": TRACE,
    })


def test_langgraph_checkpoint_is_encrypted_and_recovered_by_new_saver() -> None:
    store, cipher = asyncio.run(prepared_store())
    saver = EncryptedLangGraphCheckpointSaver(store, cipher)

    builder = StateGraph(State)
    builder.add_node("step_one", lambda state: {"value": state.get("value", 0) + 1})
    builder.add_node("step_two", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "step_two")
    builder.add_edge("step_two", END)
    graph = builder.compile(checkpointer=saver, interrupt_before=["step_two"])
    config = {"configurable": {
        "thread_id": RUN,
        "runtime_task_id": TASK,
        "runtime_trace_id": TRACE,
    }}

    first = graph.invoke({"value": 0}, config=config)
    assert first["value"] == 1

    # A second saver instance simulates a new Runtime process.  It reads the
    # encrypted envelope from the same independent store; no plaintext state
    # is placed in an ordinary Runtime record.
    restarted = StateGraph(State)
    restarted.add_node("step_one", lambda state: {"value": state.get("value", 0) + 1})
    restarted.add_node("step_two", lambda state: {"value": state["value"] + 1})
    restarted.add_edge(START, "step_one")
    restarted.add_edge("step_one", "step_two")
    restarted.add_edge("step_two", END)
    recovered_graph = restarted.compile(
        checkpointer=EncryptedLangGraphCheckpointSaver(store, cipher)
    )
    recovered = recovered_graph.invoke(None, config=config)
    assert recovered["value"] == 2

    run = asyncio.run(store.load_run(RUN))
    assert run is not None
    envelope = cipher.serialize(run.encryptedStatePayload)
    assert "value" not in envelope
    assert "sourceSampleId" not in envelope


def test_checkpoint_saver_requires_bound_opaque_runtime_ids() -> None:
    saver = EncryptedLangGraphCheckpointSaver(InMemoryRuntimeStore(), RuntimeStateCipher(b"k" * 32, "runtime-key-1"))
    with pytest.raises(Exception) as error:
        saver.get_tuple({"configurable": {"thread_id": RUN}})
    assert str(error.value) == "RUNTIME_CHECKPOINT_BINDING_INVALID"


def test_persistence_coordinator_recovers_only_typed_intent_request() -> None:
    async def scenario() -> IntentTaskRequest:
        store = InMemoryRuntimeStore()
        cipher = RuntimeStateCipher(b"k" * 32, "runtime-key-1")
        coordinator = RuntimePersistenceCoordinator(store, cipher)
        value = intent_request()
        await coordinator.begin(value)
        return await coordinator.load_intent_recovery_request(RUN)

    recovered = asyncio.run(scenario())
    assert recovered.runId == RUN
    assert recovered.taskId == TASK
    assert recovered.allowedWorkflows == ["dynamic_read_query"]
