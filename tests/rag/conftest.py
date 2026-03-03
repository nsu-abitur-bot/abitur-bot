"""
Conftest for rag tests.
Adds accuracy summary for test_keyword_in_retrieved_context after the test run.
"""

_kw_results = {"passed": 0, "xpassed": 0, "xfailed": 0, "failed": 0}


def pytest_configure(config):
    _kw_results.update({"passed": 0, "xpassed": 0, "xfailed": 0, "failed": 0})


def pytest_runtest_logreport(report):
    if report.when != "call":
        return
    if "test_keyword_in_retrieved_context" not in report.nodeid:
        return

    results = _kw_results

    if report.passed:
        if hasattr(report, "wasxfail"):
            results["xpassed"] += 1
        else:
            results["passed"] += 1
    elif report.skipped and hasattr(report, "wasxfail"):
        results["xfailed"] += 1
    elif report.failed:
        results["failed"] += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    r = _kw_results

    passed = r["passed"] + r["xpassed"]
    failed = r["failed"]
    xfailed = r["xfailed"]
    total = passed + failed + xfailed

    if total == 0:
        return

    strict_total = passed + failed          # без ожидаемых xfail
    strict_acc = (passed / strict_total * 100) if strict_total else 0.0
    overall_acc = passed / total * 100

    terminalreporter.write_sep("=", "Keyword Matching Accuracy")
    terminalreporter.write_line(f"  Passed          : {r['passed']}")
    if r["xpassed"]:
        terminalreporter.write_line(f"  Unexpected pass : {r['xpassed']} (xpass)")
    terminalreporter.write_line(f"  Expected fail   : {xfailed} (xfail)")
    terminalreporter.write_line(f"  Failed          : {failed}")
    terminalreporter.write_line(f"  Total cases     : {total}")
    terminalreporter.write_sep("-", "")
    terminalreporter.write_line(
        f"  Accuracy (excl. xfail) : {strict_acc:.1f}%  ({passed}/{strict_total})"
    )
    terminalreporter.write_line(
        f"  Accuracy (all cases)   : {overall_acc:.1f}%  ({passed}/{total})"
    )
