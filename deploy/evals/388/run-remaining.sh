#!/bin/sh
# Credentials are passed on stdin by an explicitly authorized caller.
# No credential files are opened here. Only the with-kb test project is managed.
set +x
set -eu
umask 077
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ] || [ "$1" != --execute ]; then
    printf '%s\n' 'Expected --execute. No services were started.' >&2
    exit 2
fi
shift
report_directory=remaining-eight-01
if [ "$#" -eq 1 ]; then
    if [ "$1" != --continue-after-disconnect ]; then exit 2; fi
    report_directory=remaining-eight-02
fi
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
if [ "$base" != /home/mihail/abitur-evals/388-20261107 ]; then
    printf '%s\n' 'Unexpected evaluation directory.' >&2
    exit 1
fi
command -v timeout >/dev/null
exec 9>"$base/.index-one.lock"
if ! flock -n 9; then
    printf '%s\n' 'Another pilot owns the lock.' >&2
    exit 1
fi
if [ -e "/home/mihail/abitur-evals/388-reports/with-kb/$report_directory" ]; then
    printf '%s\n' 'Prior batch exists. Inspect before retry.' >&2
    exit 1
fi
state=$(sudo -n /usr/local/sbin/abitur-eval388 with-kb status </dev/null)
if printf '%s\n' "$state" | awk 'NR>1 && NF {found=1} END {exit !found}'; then
    printf '%s\n' 'Existing test containers found; no ownership was taken.' >&2
    exit 1
fi
started=0
cleanup() {
    code=$?
    trap - 0 HUP INT TERM
    if [ "$started" -eq 1 ]; then
        if ! sudo -n /usr/local/sbin/abitur-eval388 with-kb stop </dev/null; then
            printf '%s\n' 'Test cleanup failed. Check with-kb status manually.' >&2
            if [ "$code" -eq 0 ]; then code=1; fi
        fi
    fi
    exit "$code"
}
trap cleanup 0
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
# Arm cleanup before start: a partially failed start also needs cleanup.
started=1
sudo -n /usr/local/sbin/abitur-eval388 with-kb start </dev/null
timeout --signal=TERM --kill-after=30s 15000s \
    sudo -n /usr/local/sbin/abitur-eval388 with-kb run \
    uv run --no-sync python /evaluation/admission_index_remaining.py "$@"
