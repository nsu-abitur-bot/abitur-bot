#!/bin/sh
# Checks only the staged files. It does not start containers or call providers.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$root"
test "$(id -un)" = mihail
test "$root" = /home/mihail/abitur-evals/388-20261107
test -z "$(find . -type l -print -quit)"
sha256sum -c SHA256SUMS
for variant in with-kb empty-kb; do
    for store in postgres redis lightrag faq-cache logs; do
        dir="$root/$variant/$store"
        test -d "$dir"
        test -z "$(find "$dir" -mindepth 1 -print -quit)"
    done
done
printf '\nBoth storage trees are empty. No runtime is provisioned yet.\n'
find with-kb empty-kb -maxdepth 1 -type d -print
printf '\nDirectory owner and mode:\n'
stat -c '%U %a %n' . with-kb empty-kb
