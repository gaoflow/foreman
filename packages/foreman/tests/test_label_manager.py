"""Tests for :mod:`foreman.label_manager`.

The manager is the runtime layer for the D1 typed-label catalog. These
tests cover the invariants the manager enforces, the structured
logging, the no-op semantics (post-invariant ``final == current``),
and the grep-fence that pins the single-import-point contract.

The #303 regression test is the load-bearing one: writing
``IMPL_APPROVED`` against an issue carrying a stale ``MERGING_PLAN``
strips ``MERGING_PLAN`` automatically. That's the failure class the
manager exists to eliminate.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from foreman.label_manager import (
    IssueLabelWriter,
    LabelManager,
    LabelWriter,
    ReconcilerHostLabelWriter,
    _LABEL_WRITER_GREP_ALLOWLIST,
    _apply_invariants,
    _classify,
)
from foreman.labels import Label, LabelClass


# ---------------------------------------------------------------------------
# Fake writer — single set[str], records replace_labels calls.
# ---------------------------------------------------------------------------


class FakeLabelWriter:
    """Minimal :class:`LabelWriter` for unit tests. Holds the
    'remote' label set as a single :class:`set[str]` and records every
    :meth:`replace_labels` call for assertion."""

    def __init__(self, initial: set[str] | None = None, *, url: str = "https://example.test/issues/1") -> None:
        self._labels: set[str] = set(initial or set())
        self._url = url
        self.replace_calls: list[set[str]] = []

    @property
    def issue_url(self) -> str:
        return self._url

    def read_current(self) -> set[str]:
        return set(self._labels)

    def replace_labels(self, final: set[str]) -> None:
        self.replace_calls.append(set(final))
        self._labels = set(final)


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_queue_label(self) -> None:
        assert _classify("foreman:impl-approved") is LabelClass.QUEUE

    def test_in_flight_label(self) -> None:
        assert _classify("foreman:merging-plan") is LabelClass.IN_FLIGHT

    def test_blocking_label(self) -> None:
        assert _classify("foreman:needs-help") is LabelClass.BLOCKING

    def test_counter_label(self) -> None:
        assert _classify("foreman:impl-attempt-1") is LabelClass.COUNTER

    def test_terminal_label(self) -> None:
        assert _classify("foreman:done") is LabelClass.TERMINAL

    def test_unknown_foreman_label(self) -> None:
        assert _classify("foreman:bogus-unknown") is None

    def test_non_foreman_label(self) -> None:
        assert _classify("priority:high") is None


# ---------------------------------------------------------------------------
# _apply_invariants — pure helper, no I/O.
# ---------------------------------------------------------------------------


class TestApplyInvariantsQueueStripsInFlight:
    def test_queue_add_strips_in_flight(self) -> None:
        current = {"foreman:planning", "foreman:merging-plan"}
        final, stripped_in, stripped_term = _apply_invariants(
            current, {"foreman:impl-approved"}, set()
        )
        assert final == {"foreman:impl-approved"}
        assert stripped_in == {"foreman:planning", "foreman:merging-plan"}
        assert stripped_term == set()

    def test_queue_add_strips_in_flight_preserves_non_foreman(self) -> None:
        current = {"foreman:planning", "foreman:merging-plan", "priority:high"}
        final, _, _ = _apply_invariants(
            current, {"foreman:impl-approved"}, set()
        )
        assert final == {"foreman:impl-approved", "priority:high"}

    def test_in_flight_add_does_not_strip_queue(self) -> None:
        # Inverse direction: writing IN_FLIGHT alongside a QUEUE label
        # leaves the QUEUE label alone (this is the normal
        # `plan-approved → merging-plan` transition).
        current = {"foreman:plan-approved"}
        final, stripped_in, stripped_term = _apply_invariants(
            current, {"foreman:merging-plan"}, set()
        )
        assert final == {"foreman:plan-approved", "foreman:merging-plan"}
        assert stripped_in == set()
        assert stripped_term == set()


class TestApplyInvariantsTerminalStripsNonTerminal:
    def test_terminal_add_strips_all_non_terminal_foreman_labels(self) -> None:
        current = {
            "foreman:impl-approved",
            "foreman:merging-impl",
            "foreman:impl-attempt-1",
            "priority:high",
        }
        final, stripped_in, stripped_term = _apply_invariants(
            current, {"foreman:done"}, set()
        )
        assert final == {"foreman:done", "priority:high"}
        assert stripped_term == {
            "foreman:impl-approved",
            "foreman:merging-impl",
            "foreman:impl-attempt-1",
        }
        # stripped_in is only populated by the QUEUE invariant.
        assert stripped_in == set()

    def test_terminal_add_strips_unknown_foreman_label(self) -> None:
        # Unknown foreman:* labels classify as None (not TERMINAL);
        # the TERMINAL invariant strips them defensively.
        current = {"foreman:bogus-unknown", "foreman:done-but-extra"}
        final, _, stripped_term = _apply_invariants(
            current, {"foreman:done"}, set()
        )
        assert final == {"foreman:done"}
        assert stripped_term == {"foreman:bogus-unknown", "foreman:done-but-extra"}


class TestApplyInvariantsIdempotence:
    def test_repeating_add_is_noop(self) -> None:
        current = {"foreman:plan-approved"}
        final, stripped_in, stripped_term = _apply_invariants(
            current, {"foreman:plan-approved"}, set()
        )
        assert final == current
        assert stripped_in == set()
        assert stripped_term == set()

    def test_remove_absent_label_is_noop(self) -> None:
        current = {"foreman:plan-approved"}
        final, _, _ = _apply_invariants(current, set(), {"foreman:never-here"})
        assert final == current


class TestApplyInvariantsNonForeman:
    def test_non_foreman_labels_passthrough_on_queue_add(self) -> None:
        current = {"priority:high", "needs:design", "bug"}
        final, _, _ = _apply_invariants(
            current, {"foreman:plan-approved"}, set()
        )
        assert final == {"priority:high", "needs:design", "bug", "foreman:plan-approved"}


# ---------------------------------------------------------------------------
# LabelManager.transition — orchestration, logging, write-suppression.
# ---------------------------------------------------------------------------


class TestTransitionWrite:
    def test_write_happens_when_final_differs(self) -> None:
        writer = FakeLabelWriter(initial={"foreman:planning"})
        manager = LabelManager()
        result = manager.transition(
            writer,
            add={Label.PLAN_APPROVED},
            remove={Label.PLANNING},
            reason="reviewer_clean",
        )
        assert result == {"foreman:plan-approved"}
        assert writer.replace_calls == [{"foreman:plan-approved"}]

    def test_returns_final_set(self) -> None:
        writer = FakeLabelWriter(initial=set())
        manager = LabelManager()
        result = manager.transition(
            writer,
            add={Label.PLAN_APPROVED},
            reason="test",
        )
        assert result == {"foreman:plan-approved"}

    def test_dynamic_counter_string_accepted(self) -> None:
        writer = FakeLabelWriter(initial=set())
        manager = LabelManager()
        result = manager.transition(
            writer,
            add={"foreman:fix-attempt-1"},
            reason="fixer_attempt_stamp",
        )
        assert "foreman:fix-attempt-1" in result


class TestTransitionNoop:
    def test_true_noop_skips_write_and_returns_current(self) -> None:
        writer = FakeLabelWriter(initial={"foreman:impl-approved"})
        manager = LabelManager()
        result = manager.transition(
            writer,
            add={Label.IMPL_APPROVED},
            reason="redundant_retry",
        )
        # final == current after invariants → no write.
        assert result == {"foreman:impl-approved"}
        assert writer.replace_calls == []

    def test_noop_logs_at_debug_not_info(self, caplog: pytest.LogCaptureFixture) -> None:
        writer = FakeLabelWriter(initial={"foreman:impl-approved"})
        manager = LabelManager()
        with caplog.at_level(logging.DEBUG, logger="foreman.label_manager"):
            manager.transition(
                writer, add={Label.IMPL_APPROVED}, reason="redundant_retry"
            )
        info_records = [r for r in caplog.records if r.levelno >= logging.INFO]
        assert info_records == []
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debug_records) == 1


class TestTransitionRegression303:
    """The load-bearing regression: writing IMPL_APPROVED against an
    issue still carrying a stale MERGING_PLAN must strip
    MERGING_PLAN as part of the same atomic transition."""

    def test_impl_approved_strips_stale_merging_plan(self) -> None:
        # COUNTER labels (impl-attempt-N) are NOT in-flight markers;
        # they're history. The QUEUE invariant only targets IN_FLIGHT.
        writer = FakeLabelWriter(
            initial={"foreman:merging-plan", "foreman:impl-attempt-1"}
        )
        manager = LabelManager()
        result = manager.transition(
            writer, add={Label.IMPL_APPROVED}, reason="reviewer_clean"
        )
        assert result == {"foreman:impl-approved", "foreman:impl-attempt-1"}
        assert writer.replace_calls == [
            {"foreman:impl-approved", "foreman:impl-attempt-1"}
        ]

    def test_impl_approved_strips_merging_plan_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        writer = FakeLabelWriter(initial={"foreman:merging-plan"})
        manager = LabelManager()
        with caplog.at_level(logging.DEBUG, logger="foreman.label_manager"):
            manager.transition(
                writer, add={Label.IMPL_APPROVED}, reason="reviewer_clean"
            )
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 1
        assert getattr(warns[0], "stale_in_flight_stripped", None) == [
            "foreman:merging-plan"
        ]
        assert getattr(warns[0], "reason", None) == "reviewer_clean"

    def test_idempotent_retry_still_strips_stale_label(self) -> None:
        """Post-invariant idempotence guard. ``add ⊆ current`` would
        be classified no-op by a surface-level check, but the
        QUEUE-strips-IN_FLIGHT invariant strips ``merging-plan`` so
        ``final != current`` and the write MUST fire."""
        writer = FakeLabelWriter(
            initial={
                "foreman:impl-approved",  # already there
                "foreman:merging-plan",  # stale, must be stripped
                "priority:high",
            }
        )
        manager = LabelManager()
        result = manager.transition(
            writer, add={Label.IMPL_APPROVED}, reason="reviewer_retry"
        )
        assert result == {"foreman:impl-approved", "priority:high"}
        assert writer.replace_calls == [{"foreman:impl-approved", "priority:high"}]


class TestTransitionLoggingPayload:
    def test_info_log_carries_full_extra_payload(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        writer = FakeLabelWriter(
            initial={"foreman:planning"}, url="https://github.com/x/y/issues/42"
        )
        manager = LabelManager()
        with caplog.at_level(logging.INFO, logger="foreman.label_manager"):
            manager.transition(
                writer,
                add={Label.PLAN_APPROVED},
                remove={Label.PLANNING},
                reason="reviewer_clean",
            )
        info = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info) == 1
        rec = info[0]
        assert getattr(rec, "issue_url") == "https://github.com/x/y/issues/42"
        assert getattr(rec, "from_labels") == ["foreman:planning"]
        assert getattr(rec, "to_labels") == ["foreman:plan-approved"]
        assert getattr(rec, "added") == ["foreman:plan-approved"]
        assert getattr(rec, "removed") == ["foreman:planning"]
        assert getattr(rec, "reason") == "reviewer_clean"


class TestTransitionTerminalInvariant:
    def test_done_strips_all_non_terminal_foreman(self) -> None:
        writer = FakeLabelWriter(
            initial={
                "foreman:impl-approved",
                "foreman:merging-impl",
                "foreman:impl-attempt-1",
                "priority:high",
            }
        )
        manager = LabelManager()
        result = manager.transition(
            writer, add={Label.DONE}, reason="autonomous_close"
        )
        assert result == {"foreman:done", "priority:high"}


# ---------------------------------------------------------------------------
# Grep fence — the single-import-point enforcement.
# ---------------------------------------------------------------------------


_FORBIDDEN_PYGITHUB_METHODS = frozenset(
    {"add_to_labels", "remove_from_labels", "set_labels"}
)


class _PyGithubLabelCallFinder(ast.NodeVisitor):
    """AST visitor that flags ``obj.<method>(...)`` calls where
    ``<method>`` is a PyGithub label-write method.

    Crucially, we only flag method *calls* (``Call(func=Attribute)``).
    Method *definitions* (``def set_labels(self, ...): ...``) and bare
    identifier uses are NOT flagged — those are the project's own
    abstraction (e.g. ``ReconcilerHost.set_labels``), not the PyGithub
    call surface.
    """

    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_PYGITHUB_METHODS:
            self.hits.append((node.lineno, func.attr))
        self.generic_visit(node)


class TestLabelWritesOnlyGoThroughLabelManager:
    """The single-import-point enforcement. Only :mod:`label_manager`
    + the two thin PyGithub wrappers in :mod:`daemon_host` and
    :mod:`git_hosts.github` may call PyGithub's label-write
    methods.

    Uses AST rather than substring search so we don't trip on the
    project's OWN ``def set_labels(self, ...)`` methods (e.g. on
    :class:`~foreman.reconciler.host.ReconcilerHost`) — those are
    intentional abstractions that the manager itself routes through.
    Only ``obj.set_labels(...)`` *invocations* count as PyGithub-side
    label writes.
    """

    def test_no_pygithub_label_call_outside_allowlist(self) -> None:
        src_root = (
            Path(__file__).resolve().parent.parent / "src" / "foreman"
        )
        assert src_root.is_dir(), f"src root not found: {src_root}"

        offenders: list[tuple[str, int, str]] = []
        for path in src_root.rglob("*.py"):
            rel_to_pkg = path.relative_to(src_root).as_posix()
            allowlist_key = f"foreman/{rel_to_pkg}"
            if allowlist_key in _LABEL_WRITER_GREP_ALLOWLIST:
                continue

            tree = ast.parse(path.read_text(encoding="utf-8"))
            finder = _PyGithubLabelCallFinder()
            finder.visit(tree)
            for lineno, attr in finder.hits:
                offenders.append((allowlist_key, lineno, attr))

        assert offenders == [], (
            "Files outside the allowlist must not call PyGithub's label-write "
            "methods (add_to_labels / remove_from_labels / set_labels). Route "
            "through LabelManager().transition(...) instead.\n"
            f"Offenders (file, line, method): {offenders}"
        )


# ---------------------------------------------------------------------------
# Adapter smoke tests — verify URL construction + delegation shape.
# ---------------------------------------------------------------------------


class _FakeIssue:
    """Mimics the parts of PyGithub ``Issue`` the writer adapter
    uses."""

    def __init__(self, html_url: str, labels: list[str]) -> None:
        self.html_url = html_url
        self._labels_data = list(labels)
        self.updated = False
        self.set_labels_calls: list[tuple[str, ...]] = []

    def update(self) -> None:
        self.updated = True

    @property
    def labels(self):
        return [type("_L", (), {"name": n})() for n in self._labels_data]

    def set_labels(self, *names: str) -> None:
        self.set_labels_calls.append(names)
        self._labels_data = list(names)


class TestIssueLabelWriter:
    def test_issue_url_sources_from_html_url(self) -> None:
        issue = _FakeIssue("https://github.com/o/r/issues/9", [])
        writer = IssueLabelWriter(issue)
        assert writer.issue_url == "https://github.com/o/r/issues/9"

    def test_read_current_invalidates_cache(self) -> None:
        issue = _FakeIssue("u", ["foreman:planning"])
        writer = IssueLabelWriter(issue)
        result = writer.read_current()
        assert issue.updated is True
        assert result == {"foreman:planning"}

    def test_replace_labels_sorted(self) -> None:
        issue = _FakeIssue("u", [])
        writer = IssueLabelWriter(issue)
        writer.replace_labels({"foreman:b", "foreman:a"})
        assert issue.set_labels_calls == [("foreman:a", "foreman:b")]


class _FakeReconcilerHost:
    def __init__(self) -> None:
        self.get_calls: list[tuple[str, str, int]] = []
        self.set_calls: list[tuple[str, str, int, set[str]]] = []
        self._labels: set[str] = set()

    def get_labels(self, *, owner: str, repo: str, issue: int) -> set[str]:
        self.get_calls.append((owner, repo, issue))
        return set(self._labels)

    def set_labels(
        self, *, owner: str, repo: str, issue: int, labels: set[str]
    ) -> None:
        self.set_calls.append((owner, repo, issue, set(labels)))
        self._labels = set(labels)


class TestReconcilerHostLabelWriter:
    def test_issue_url_constructed_from_triple(self) -> None:
        host = _FakeReconcilerHost()
        writer = ReconcilerHostLabelWriter(
            host, owner="acme", repo="widget", issue=42
        )
        assert writer.issue_url == "https://github.com/acme/widget/issues/42"

    def test_delegates_get_and_set(self) -> None:
        host = _FakeReconcilerHost()
        writer = ReconcilerHostLabelWriter(host, owner="o", repo="r", issue=1)
        writer.replace_labels({"foreman:plan-approved"})
        assert host.set_calls == [("o", "r", 1, {"foreman:plan-approved"})]
        assert writer.read_current() == {"foreman:plan-approved"}
        assert host.get_calls == [("o", "r", 1)]


# ---------------------------------------------------------------------------
# Protocol structural conformance — the runtime check is implicit,
# but pinning the signatures here catches accidental drift.
# ---------------------------------------------------------------------------


def test_fake_writer_satisfies_protocol() -> None:
    writer: LabelWriter = FakeLabelWriter()
    assert callable(writer.read_current)
    assert callable(writer.replace_labels)
    assert isinstance(writer.issue_url, str)
