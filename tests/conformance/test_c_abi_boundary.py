"""
No exception may cross the C ABI.

``extern "C"`` says nothing about unwinding, so every entry point is a
function-try-block whose handler returns that function's "cannot answer" value.
This reads the sources rather than running anything: the property is about
every entry point, not the ones a test happens to reach.

Two are exempt -- returning a preprocessor constant, and returning a string
literal from a switch over an enum -- because neither can allocate.
"""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "cpp" / "include" / "pvzstd" / "pvzstd.h"
SOURCES = sorted((ROOT / "cpp" / "src").glob("*.cpp"))

CANNOT_THROW = {
    "pvzstd_abi_version": "returns a preprocessor constant",
    "pvzstd_max_file_version": "returns a preprocessor constant",
    "pvzstd_status_message": "returns a string literal from a switch over an enum",
}


def _declared() -> list[str]:
    return sorted(set(re.findall(r"PVZSTD_API\s+[\w *]+?\b(pvzstd_\w+)\s*\(", HEADER.read_text())))


def _definition(name: str) -> tuple[Path, bool] | None:
    """Locate a definition, and report whether it is a function-try-block."""
    pattern = re.compile(rf"^[\w *]*\b{re.escape(name)}\s*\([^)]*\)\s*(try)?\s*\{{", re.MULTILINE | re.DOTALL)
    for source in SOURCES:
        found = pattern.search(source.read_text())
        if found:
            return source, bool(found.group(1))
    return None


def test_every_entry_point_is_defined() -> None:
    """A declared symbol with no definition is a link error waiting for a caller."""
    assert HEADER.exists(), f"header not found at {HEADER}"
    declared = _declared()
    assert declared, "found no PVZSTD_API declarations; the matcher is broken, not the header"
    missing = [name for name in declared if _definition(name) is None]
    assert not missing, f"declared but not defined in cpp/src: {missing}"


def test_no_exception_can_leave_the_c_abi() -> None:
    """Every entry point catches, except the two that have nothing to catch."""
    unguarded = []
    for name in _declared():
        located = _definition(name)
        assert located is not None  # test_every_entry_point_is_defined says which
        source, guarded = located
        if not guarded and name not in CANNOT_THROW:
            unguarded.append(f"{name} ({source.relative_to(ROOT)})")

    assert not unguarded, (
        "these entry points can let an exception reach a C caller: "
        + ", ".join(unguarded)
        + ". Make each a function-try-block returning what that function already "
        "uses for 'cannot answer' -- a status, NULL, 0 or -1."
    )


def test_the_exemptions_still_exist() -> None:
    """An exemption for a function that is gone hides the next one added."""
    declared = set(_declared())
    stale = sorted(set(CANNOT_THROW) - declared)
    assert not stale, f"CANNOT_THROW names entry points that no longer exist: {stale}"
