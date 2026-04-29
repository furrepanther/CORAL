"""Tests for the manager's earlier stall detection (issue #73).

Covers the two pure-function-ish helpers that gate the new logic:
- ``AgentManager._classify_stall`` — tier classifier (ok / warn / stall)
- ``AgentManager._agent_has_pending_attempt`` — exempts grader-wait windows

The full monitor_loop integration is harder to unit-test (threads, real
processes); these helpers are where the new behavior actually lives.
"""

from __future__ import annotations

import json
from pathlib import Path

from coral.agent.manager import AgentManager
from coral.config import CoralConfig
from coral.workspace import ProjectPaths


def _build_manager(tmp_path: Path) -> tuple[AgentManager, ProjectPaths]:
    coral_dir = tmp_path / ".coral"
    (coral_dir / "public" / "attempts").mkdir(parents=True)
    (coral_dir / "public" / "logs").mkdir()

    paths = ProjectPaths(
        results_dir=tmp_path / "results",
        task_dir=tmp_path,
        run_dir=tmp_path,
        coral_dir=coral_dir,
        agents_dir=tmp_path / "agents",
        repo_dir=tmp_path / "repo",
    )
    cfg = CoralConfig.from_dict({
        "task": {"name": "t", "description": "d"},
        "agents": {"runtime": "claude-code"},
    })
    manager = AgentManager(cfg, verbose=False)
    manager.paths = paths
    return manager, paths


# --------------------------------------------------------------------------- #
# _classify_stall — tier boundaries                                           #
# --------------------------------------------------------------------------- #

def test_classify_stall_below_warn_returns_ok():
    assert AgentManager._classify_stall(age=0, warn_after=600, restart_after=1800) == "ok"
    assert AgentManager._classify_stall(age=599, warn_after=600, restart_after=1800) == "ok"


def test_classify_stall_at_warn_threshold_returns_warn():
    assert AgentManager._classify_stall(age=600, warn_after=600, restart_after=1800) == "warn"
    assert AgentManager._classify_stall(age=1799, warn_after=600, restart_after=1800) == "warn"


def test_classify_stall_at_restart_threshold_returns_stall():
    assert AgentManager._classify_stall(age=1800, warn_after=600, restart_after=1800) == "stall"
    assert AgentManager._classify_stall(age=10000, warn_after=600, restart_after=1800) == "stall"


def test_classify_stall_warn_disabled_skips_warn_tier():
    """warn_after=0 disables the warning; only the stall tier remains."""
    assert AgentManager._classify_stall(age=1000, warn_after=0, restart_after=1800) == "ok"
    assert AgentManager._classify_stall(age=1799, warn_after=0, restart_after=1800) == "ok"
    assert AgentManager._classify_stall(age=1800, warn_after=0, restart_after=1800) == "stall"


def test_classify_stall_disabled_when_restart_after_is_zero():
    """restart_after<=0 disables stall detection entirely (current opt-out)."""
    assert AgentManager._classify_stall(age=1_000_000, warn_after=600, restart_after=0) == "ok"
    assert AgentManager._classify_stall(age=1_000_000, warn_after=600, restart_after=-1) == "ok"


def test_classify_stall_handles_warn_above_restart():
    """Misconfiguration (warn_after >= restart_after): stall still wins at age >= restart_after."""
    assert AgentManager._classify_stall(age=1800, warn_after=2000, restart_after=1800) == "stall"
    # Below the stall threshold, the warning never fires because the warn
    # threshold is higher than the stall threshold.
    assert AgentManager._classify_stall(age=1799, warn_after=2000, restart_after=1800) == "ok"


# --------------------------------------------------------------------------- #
# _agent_has_pending_attempt — gates the stall check                          #
# --------------------------------------------------------------------------- #

def test_agent_has_pending_attempt_true_when_pending_exists(tmp_path: Path) -> None:
    """A pending attempt for the agent should mark them as in-eval."""
    manager, paths = _build_manager(tmp_path)
    attempts_dir = paths.coral_dir / "public" / "attempts"
    (attempts_dir / "abc.json").write_text(
        json.dumps({"agent_id": "agent-1", "status": "pending"})
    )
    assert manager._agent_has_pending_attempt("agent-1") is True


def test_agent_has_pending_attempt_false_for_other_agents(tmp_path: Path) -> None:
    """Pending attempts for other agents must not exempt this one."""
    manager, paths = _build_manager(tmp_path)
    attempts_dir = paths.coral_dir / "public" / "attempts"
    (attempts_dir / "abc.json").write_text(
        json.dumps({"agent_id": "agent-2", "status": "pending"})
    )
    assert manager._agent_has_pending_attempt("agent-1") is False


def test_agent_has_pending_attempt_false_when_only_scored(tmp_path: Path) -> None:
    """Already-scored attempts don't exempt the agent — they're done waiting."""
    manager, paths = _build_manager(tmp_path)
    attempts_dir = paths.coral_dir / "public" / "attempts"
    (attempts_dir / "abc.json").write_text(
        json.dumps({"agent_id": "agent-1", "status": "improved"})
    )
    (attempts_dir / "def.json").write_text(
        json.dumps({"agent_id": "agent-1", "status": "regressed"})
    )
    assert manager._agent_has_pending_attempt("agent-1") is False


def test_agent_has_pending_attempt_handles_malformed_files(tmp_path: Path) -> None:
    """Garbage JSON (e.g. mid-rename) must not raise."""
    manager, paths = _build_manager(tmp_path)
    attempts_dir = paths.coral_dir / "public" / "attempts"
    (attempts_dir / "garbage.json").write_text("{not valid json")
    (attempts_dir / "real.json").write_text(
        json.dumps({"agent_id": "agent-1", "status": "pending"})
    )
    assert manager._agent_has_pending_attempt("agent-1") is True


def test_agent_has_pending_attempt_false_when_no_attempts_dir(tmp_path: Path) -> None:
    """Brand-new run with no attempts dir doesn't crash."""
    manager, paths = _build_manager(tmp_path)
    # Remove the attempts dir to simulate the empty-state.
    import shutil
    shutil.rmtree(paths.coral_dir / "public" / "attempts")
    assert manager._agent_has_pending_attempt("agent-1") is False
