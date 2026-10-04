from knowledge_graph.experiments.telemetry import (
    QueryTelemetry,
    bind_telemetry,
    record_stage0_policy,
)


def _record(*, proposed: bool, allowed: bool, stress: bool) -> QueryTelemetry:
    telemetry = QueryTelemetry(query_id="q0")
    telemetry.start()
    with bind_telemetry(telemetry):
        telemetry.total_llm_calls = 2
        telemetry.answer_calls = 1
        telemetry.auxiliary_calls = 1
        telemetry.input_tokens = 123
        telemetry.output_tokens = 17
        record_stage0_policy(
            "Paris",
            proposed_final=proposed,
            allow_early_exit=allowed,
            reasoning_stress=stress,
        )
    return telemetry


def test_stage0_exit_is_accepted_only_in_enabled_non_stress_arm():
    accepted = _record(proposed=True, allowed=True, stress=False)
    assert accepted.stage0_policy == "early_exit_enabled"
    assert accepted.stage0_candidate_answer == "Paris"
    assert accepted.stage0_early_exit
    assert accepted.stage0_rejection_reason == ""
    assert accepted.stage0_total_llm_calls == 2
    assert accepted.stage0_answer_calls == 1
    assert accepted.stage0_auxiliary_calls == 1
    assert accepted.stage0_input_tokens == 123
    assert accepted.stage0_output_tokens == 17
    assert accepted.stage0_elapsed_s >= 0


def test_force_continue_preserves_candidate_but_rejects_exit():
    rejected = _record(proposed=True, allowed=False, stress=False)
    assert rejected.stage0_policy == "force_continue"
    assert rejected.stage0_candidate_answer == "Paris"
    assert not rejected.stage0_early_exit
    assert rejected.stage0_rejection_reason == "stage0_exit_disabled"


def test_nonfinal_stage0_decision_is_not_counted_as_early_exit():
    continued = _record(proposed=False, allowed=True, stress=False)
    assert continued.stage0_candidate_answer == ""
    assert not continued.stage0_early_exit
    assert continued.stage0_rejection_reason == "model_requested_graph_expansion"
