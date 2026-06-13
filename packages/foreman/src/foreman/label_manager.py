"""Single owner for ``foreman:*`` label lifecycle transitions.

This module is the runtime layer the D1 typed-label catalog
(:mod:`foreman.labels`) was waiting for. It owns every write to the
``foreman:*`` namespace and enforces classification-level invariants:

- Writing a :attr:`~foreman.labels.LabelClass.QUEUE` label (e.g.
  ``foreman:impl-approved``) strips every
  :attr:`~foreman.labels.LabelClass.IN_FLIGHT` label currently on the
  issue in the SAME transition. This is the regression case from
  issue #303 — `merging-plan` outliving the spec-PR merge silently
  disables downstream rule predicates that gate on
  ``"foreman:merging-plan" not in ctx.issue.labels``.
- Writing a :attr:`~foreman.labels.LabelClass.TERMINAL` label
  (``foreman:done``, ``foreman:failed``) strips every non-terminal
  ``foreman:*`` label.
- Non-``foreman:*`` labels (``priority:high``, ``needs:design``, etc.)
  pass through untouched.

Every transition emits exactly ONE structured log line: INFO when the
write fires, DEBUG when the transition is a true no-op (``final ==
current`` after invariants), WARNING (additionally) when an invariant
strips stale labels. Stale-label rot becomes loud, not silent.

Callers depend on the narrow :meth:`LabelManager.transition` API and a
small :class:`LabelWriter` protocol. The two adapters
(:class:`IssueLabelWriter`, :class:`ReconcilerHostLabelWriter`) make
PyGithub and reconciler-host call sites interchangeable.

This module is the ONLY non-adapter site allowed to write
``foreman:*`` labels — see :data:`_LABEL_WRITER_GREP_ALLOWLIST` and the
``test_label_writes_only_go_through_label_manager`` enforcement test.
"""

from __future__ import annotations

import logging
from typing import Protocol

from foreman.labels import (
    BLOCKING_LABELS,
    COUNTER_LABELS,
    IN_FLIGHT_LABELS,
    QUEUE_LABELS,
    TERMINAL_LABELS,
    Label,
    LabelClass,
)

logger = logging.getLogger("foreman.label_manager")


_LABEL_WRITER_GREP_ALLOWLIST: frozenset[str] = frozenset(
    {
        "foreman/label_manager.py",
        "foreman/daemon_host.py",
        "foreman/git_hosts/github.py",
    }
)
"""Files allowed to reference PyGithub's ``add_to_labels`` /
``remove_from_labels`` / ``set_labels`` symbols.

Enforced by ``tests/test_label_manager.py::
test_label_writes_only_go_through_label_manager``. The manager owns
the logic; the adapter wrappers in ``daemon_host.py`` and
``git_hosts/github.py`` perform the PyGithub call.
"""


class LabelWriter(Protocol):
    """The narrow interface :class:`LabelManager` writes through.

    Two concrete adapters ship alongside the manager
    (:class:`IssueLabelWriter` wraps a PyGithub ``Issue``;
    :class:`ReconcilerHostLabelWriter` wraps a reconciler host +
    owner/repo/issue triple). Either can satisfy the contract; the
    manager doesn't care which.

    ``issue_url`` lets the manager populate the ``extra={"issue_url":
    ...}`` log payload without the caller having to thread it through
    — keeps the "manager computes everything" contract intact.
    """

    @property
    def issue_url(self) -> str:
        """HTTPS URL of the GitHub issue being mutated. Used only for
        log context."""
        ...

    def read_current(self) -> set[str]:
        """Return the live label-name set on the issue. The manager
        treats this as authoritative — any cached state inside the
        writer should be invalidated before this returns."""
        ...

    def replace_labels(self, final: set[str]) -> None:
        """Atomically replace the issue's label set with ``final``.
        Equivalent to PyGithub's ``Issue.set_labels(*sorted(final))``
        — one PUT, not two."""
        ...


def _classify(label_str: str) -> LabelClass | None:
    """Resolve a ``foreman:*`` label string to its
    :class:`~foreman.labels.LabelClass`, or ``None`` if it doesn't
    belong to any known frozenset.

    The lookup uses the D1 ``*_LABELS`` frozensets, NOT the
    :class:`Label` enum. Strings outside the enum (e.g. a hypothetical
    ``foreman:impl-attempt-99`` beyond the enumerated counters) still
    classify correctly if they're in the relevant frozenset; the
    ``None`` return signals "unknown ``foreman:*`` label" — the manager
    logs a WARNING when it sees one, since it usually indicates a stale
    label class missing from D1.
    """
    if label_str in QUEUE_LABELS:
        return LabelClass.QUEUE
    if label_str in IN_FLIGHT_LABELS:
        return LabelClass.IN_FLIGHT
    if label_str in BLOCKING_LABELS:
        return LabelClass.BLOCKING
    if label_str in COUNTER_LABELS:
        return LabelClass.COUNTER
    if label_str in TERMINAL_LABELS:
        return LabelClass.TERMINAL
    return None


def _normalize(labels: set[Label | str]) -> set[str]:
    """Coerce a mixed-shape input set to bare strings. Accepts
    :class:`Label` enum members (StrEnum — ``str(member)`` returns
    the value) and bare strings (the dynamic
    ``foreman:fix-attempt-N`` shape)."""
    return {str(label) for label in labels}


def _apply_invariants(
    current: set[str],
    add: set[str],
    remove: set[str],
) -> tuple[set[str], set[str], set[str]]:
    """Pure helper. Compute the final label set + the two sets of
    invariant-stripped labels.

    Returns ``(final, stripped_in_flight, stripped_non_terminal)``.

    The base composition is ``final = (current | add) - remove``. The
    invariants then strip additional labels from ``final``:

    - If any label in ``add`` is :attr:`LabelClass.QUEUE`, every
      :attr:`LabelClass.IN_FLIGHT` label currently on the issue is
      stripped. This is the #303 regression fix — writing
      ``IMPL_APPROVED`` against an issue still carrying a stale
      ``MERGING_PLAN`` ensures the rot is cleaned up.
    - If any label in ``add`` is :attr:`LabelClass.TERMINAL`, every
      non-terminal ``foreman:*`` label is stripped.

    Non-``foreman:*`` labels (``priority:high``, etc.) pass through
    untouched in all cases.
    """
    final = (current | add) - remove
    stripped_in_flight: set[str] = set()
    stripped_non_terminal: set[str] = set()

    add_classes = {_classify(label) for label in add}

    if LabelClass.QUEUE in add_classes:
        for label in current:
            if _classify(label) is LabelClass.IN_FLIGHT:
                stripped_in_flight.add(label)
                final.discard(label)

    if LabelClass.TERMINAL in add_classes:
        for label in current:
            if not label.startswith("foreman:"):
                continue
            if _classify(label) is LabelClass.TERMINAL:
                continue
            stripped_non_terminal.add(label)
            final.discard(label)

    return final, stripped_in_flight, stripped_non_terminal


class LabelManager:
    """Single owner for ``foreman:*`` label writes.

    Construct freshly per call site (the class is stateless; the
    instance shape exists to mirror the D1 catalog pattern and leaves
    room for per-instance configuration — e.g. an injected logger or
    test-mode flag — in future PRs).

    See module docstring for the full invariant + logging contract.
    """

    def transition(
        self,
        writer: LabelWriter,
        *,
        add: set[Label | str] | None = None,
        remove: set[Label | str] | None = None,
        reason: str,
    ) -> set[str]:
        """Apply the transition through ``writer``.

        Args:
            writer: The :class:`LabelWriter` (PyGithub adapter or
                reconciler-host adapter) backing this issue.
            add: Labels to add. Accepts :class:`Label` enum members
                and bare strings (the latter for dynamic
                ``foreman:fix-attempt-N`` counters). ``None`` is
                treated as the empty set.
            remove: Labels to remove. Same shape as ``add``. Removing
                a label that is already absent is a no-op (the manager
                swallows PyGithub's 404-on-missing-label at the API
                boundary).
            reason: Audit-log tag for this transition. Short kebab-
                case strings (``reviewer_clean``, ``worker_exception``,
                ``crash_revert``). Surfaces in every log line so
                stale-label rot can be traced back to the role that
                introduced it.

        Returns:
            The final label set actually applied (after invariants).
            On a no-op transition (``final == current`` post-
            invariants), returns ``current`` unchanged and does NOT
            call :meth:`LabelWriter.replace_labels`.

        The single log line emits at DEBUG when this is a true no-op
        and INFO otherwise. A WARNING (separate line) fires when an
        invariant stripped stale labels.
        """
        add_norm = _normalize(add or set())
        remove_norm = _normalize(remove or set())

        current = writer.read_current()
        final, stripped_in_flight, stripped_non_terminal = _apply_invariants(
            current, add_norm, remove_norm
        )

        added = final - current
        removed = current - final

        log_payload = {
            "issue_url": writer.issue_url,
            "from_labels": sorted(current),
            "to_labels": sorted(final),
            "added": sorted(added),
            "removed": sorted(removed),
            "reason": reason,
        }

        if stripped_in_flight or stripped_non_terminal:
            stale_payload = {**log_payload}
            if stripped_in_flight:
                stale_payload["stale_in_flight_stripped"] = sorted(stripped_in_flight)
            if stripped_non_terminal:
                stale_payload["stale_non_terminal_stripped"] = sorted(stripped_non_terminal)
            logger.warning("label_manager.stale_strip", extra=stale_payload)

        if final == current:
            logger.debug("label_manager.transition", extra=log_payload)
            return current

        logger.info("label_manager.transition", extra=log_payload)
        writer.replace_labels(final)
        return final


class IssueLabelWriter:
    """Adapts a PyGithub ``Issue`` to :class:`LabelWriter`.

    Used by role modules (Worker, Fixer, Reviewer, Planner) where the
    PyGithub ``Issue`` is already in scope. Invalidates PyGithub's
    label cache on every :meth:`read_current` so the manager observes
    the live remote state, not a stale local copy.
    """

    def __init__(self, issue) -> None:
        # No PyGithub type annotation: keeps this module importable
        # without PyGithub at typecheck-time and matches the style of
        # the existing daemon_host.py wrappers.
        self._issue = issue

    @property
    def issue_url(self) -> str:
        return self._issue.html_url

    def read_current(self) -> set[str]:
        self._issue.update()
        return {label.name for label in self._issue.labels}

    def replace_labels(self, final: set[str]) -> None:
        self._issue.set_labels(*sorted(final))


class ReconcilerHostLabelWriter:
    """Adapts a reconciler host + ``(owner, repo, issue)`` triple to
    :class:`LabelWriter`.

    Used by ``reconciler/actions.py`` handlers, which dispatch through
    the host abstraction rather than holding a PyGithub ``Issue``.
    Requires the host to implement ``get_labels(owner=, repo=,
    issue=)`` and ``set_labels(owner=, repo=, issue=, labels=)``;
    both are added to the protocol as part of this PR.
    """

    def __init__(self, host, *, owner: str, repo: str, issue: int) -> None:
        self._host = host
        self._owner = owner
        self._repo = repo
        self._issue = issue

    @property
    def issue_url(self) -> str:
        return f"https://github.com/{self._owner}/{self._repo}/issues/{self._issue}"

    def read_current(self) -> set[str]:
        return self._host.get_labels(owner=self._owner, repo=self._repo, issue=self._issue)

    def replace_labels(self, final: set[str]) -> None:
        self._host.set_labels(
            owner=self._owner, repo=self._repo, issue=self._issue, labels=final
        )


__all__ = [
    "IssueLabelWriter",
    "LabelManager",
    "LabelWriter",
    "ReconcilerHostLabelWriter",
    "_LABEL_WRITER_GREP_ALLOWLIST",
    "_apply_invariants",
    "_classify",
    "logger",
]
