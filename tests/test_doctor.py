"""Tests for doctor probes."""

import json
from pathlib import Path

from cluxion_hermes_call.doctor.framework import run_doctor
from cluxion_hermes_call.doctor.probes import PROBES


def test_new_probes_registered_and_non_skip():
    catalog_path = Path(__file__).parent.parent / "src" / "cluxion_hermes_call" / "doctor" / "catalog.json"
    # run doctor (our new probes do not need runner)
    result = run_doctor(
        cwd=Path.cwd(),
        hermes_bin="hermes",
        catalog_path=catalog_path,
        probes=PROBES,
        plugin="hermes-call",
        version="0.3.2",
    )
    check_map = {c.check_id: c for c in result.checks}
    # assert our two new probes return non-skip
    p1 = check_map.get("python_version_incompatibility")
    assert p1 is not None
    assert p1.status in ("pass", "warn")  # non-skip
    p2 = check_map.get("json_mode_output_malformed")
    assert p2 is not None
    assert p2.status in ("pass", "warn", "fail")  # non-skip
    # determinism of json output
    j1 = json.dumps(result.to_json_object(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    j2 = json.dumps(result.to_json_object(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert j1 == j2


def test_contract_probes_are_registered_or_removed():
    catalog_path = Path(__file__).parent.parent / "src" / "cluxion_hermes_call" / "doctor" / "catalog.json"
    result = run_doctor(
        cwd=Path.cwd(),
        hermes_bin="hermes",
        catalog_path=catalog_path,
        probes=PROBES,
        plugin="hermes-call",
        version="0.3.11",
    )
    check_map = {c.check_id: c for c in result.checks}
    assert [c.check_id for c in result.checks if c.severity == "critical" and c.status == "skip"] == []

    for check_id in (
        "session_cleanup_race_condition",
        "subprocess_timeout_not_enforced",
        "process_group_termination_fails",
        "sessions_list_parse_failure",
        "hermes_command_flag_incompatible",
        "model_argument_invalid",
        "jobs_root_not_writable",
    ):
        assert check_map[check_id].status != "skip"
