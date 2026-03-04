"""
Conftest for rag tests.
Adds accuracy summary for keyword matching and retrieval quality tests.
"""


def _empty_counters():
    return {"passed": 0, "xpassed": 0, "xfailed": 0, "failed": 0}


# Tracked test groups: test function name → (label, counters)
_TRACKED_TESTS = {
    "test_keyword_in_retrieved_context": ("Keyword Matching", _empty_counters()),
    "test_context_completeness":         ("Context Completeness", _empty_counters()),
    "test_ranking_top1_relevance":       ("Ranking Quality (top-1)", _empty_counters()),
    "test_cross_topic_isolation":        ("Cross-Topic Isolation", _empty_counters()),
    "test_factual_number_retrieval":     ("Factual Number Retrieval", _empty_counters()),
    "test_query_robustness":             ("Query Robustness", _empty_counters()),
}


def pytest_configure(config):
    for key in _TRACKED_TESTS:
        _TRACKED_TESTS[key] = (_TRACKED_TESTS[key][0], _empty_counters())


def pytest_runtest_logreport(report):
    if report.when != "call":
        return

    for test_name, (_, counters) in _TRACKED_TESTS.items():
        if test_name in report.nodeid:
            if report.passed:
                if hasattr(report, "wasxfail"):
                    counters["xpassed"] += 1
                else:
                    counters["passed"] += 1
            elif report.skipped and hasattr(report, "wasxfail"):
                counters["xfailed"] += 1
            elif report.failed:
                counters["failed"] += 1
            break


def _print_group(tw, label, r):
    """Выводит статистику по одной группе тестов."""
    passed = r["passed"] + r["xpassed"]
    failed = r["failed"]
    xfailed = r["xfailed"]
    total = passed + failed + xfailed

    if total == 0:
        return

    strict_total = passed + failed
    strict_acc = (passed / strict_total * 100) if strict_total else 0.0
    overall_acc = passed / total * 100

    tw.write_line(f"  {label}:")
    tw.write_line(f"    Passed: {r['passed']}  |  xfail: {xfailed}  |  Failed: {failed}  |  Total: {total}")
    if r["xpassed"]:
        tw.write_line(f"    Unexpected pass (xpass): {r['xpassed']}")
    tw.write_line(
        f"    Accuracy (excl. xfail): {strict_acc:.1f}%  ({passed}/{strict_total})"
        f"   |   Accuracy (all): {overall_acc:.1f}%  ({passed}/{total})"
    )
    tw.write_line("")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Собираем общую статистику
    total_passed = 0
    total_failed = 0
    total_xfailed = 0
    has_data = False

    for _, (_, r) in _TRACKED_TESTS.items():
        p = r["passed"] + r["xpassed"]
        total_passed += p
        total_failed += r["failed"]
        total_xfailed += r["xfailed"]
        if p + r["failed"] + r["xfailed"] > 0:
            has_data = True

    if not has_data:
        return

    tw = terminalreporter
    tw.write_sep("=", "RAG Retrieval Quality Report")
    tw.write_line("")

    for _, (label, r) in _TRACKED_TESTS.items():
        _print_group(tw, label, r)

    # Итого
    grand_total = total_passed + total_failed + total_xfailed
    strict_total = total_passed + total_failed
    strict_acc = (total_passed / strict_total * 100) if strict_total else 0.0
    overall_acc = (total_passed / grand_total * 100) if grand_total else 0.0

    tw.write_sep("-", "Overall")
    tw.write_line(
        f"  Total: {grand_total}  |  Passed: {total_passed}  |  xfail: {total_xfailed}  |  Failed: {total_failed}"
    )
    tw.write_line(
        f"  Accuracy (excl. xfail): {strict_acc:.1f}%  ({total_passed}/{strict_total})"
        f"   |   Accuracy (all): {overall_acc:.1f}%  ({total_passed}/{grand_total})"
    )
