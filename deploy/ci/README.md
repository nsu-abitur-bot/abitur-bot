# CI deployment for demo instances

Backend publication is triggered by successful `PR checks` for a push to this
repository's main branch; every checkout uses the tested `head_sha`. Frontend
publication requires both Check jobs on push main. PRs cannot publish or deploy.
Both use ephemeral GITHUB_TOKEN, GHCR immutable digest pins and OCI revision.
The backend main-tag promotion keeps its separate non-cancelling freshness lock.
A server lock serializes backend and web, including old reruns; GitHub main is
checked inside that lock and again after pull. No production builds occur.

The default remains the original NSU deployment. For multiple instances install
`deploy-targets.example.json` as `/etc/abitur-instances/deploy-targets.json`, owned
by root and not writable by group/others (recommended mode `0600`). Each entry
declares an instance name, distinct local API/UI ports and whether its Telegram
process must run. The helper derives `/usr/local/bin/abitur-<name>-compose`; paths
and URLs cannot be supplied by an Actions request. The file is limited to three
instances and 4 KiB; malformed, symlinked or untrusted files fail closed.

All wrappers use the same image-only `/opt/abitur/images.yaml`, but separate
environment files, Compose project names, credentials, networks and volumes.
An update pulls once, migrates each database once, starts the requested component
in every instance and requires every instance's HTTP health check. Bot-enabled
instances additionally require a stable running bot. A failure restores the old
component pin and restarts that component across all instances without repeating
migrations. Successful receipts include the affected instance names. Application
rollback does not reverse database migrations; back up every database first.
An exact retry is a no-op only when the recorded target configuration also
matches: adding an instance or changing its ports/bot selection reconciles the
targets even when the tested image is unchanged. Older receipts are reconciled
once and retain their previous-image rollback reference.

Set `bot: false` until the instance has its own enabled bot token. Never use one
Telegram token in two projects or leave an old server polling during cutover.
Details of host preparation and transfer: [migration runbook](../../docs/demo-host-migration.md).

## Root provisioning (outside Actions)

Copy these reviewed files from the exact merged commit, not mutable main:

```sh
install -o root -g root -m 0755 deploy/ci/abitur_deploy.py /usr/local/sbin/abitur-deploy
install -o root -g root -m 0755 deploy/ci/abitur-ci-ssh /usr/local/sbin/abitur-ci-ssh
```

Requires Python 3 stdlib, Docker Compose, existing
`/usr/local/bin/abitur-nsu-compose`, and a root-owned `/opt/abitur` directory.
Keep existing compose wrapper flags/env/base/`minimal.yaml`; append the third
`-f /opt/abitur/images.yaml` overlay. Initialize the image-only JSON overlay:

```json
{"services":{"api":{"image":"abitur-backend:main-a98dea2"},"bot":{"image":"abitur-backend:main-a98dea2"},"migrate":{"image":"abitur-backend:main-a98dea2"},"permissions":{"image":"abitur-backend:main-a98dea2"},"web":{"image":"abitur-nsu-web:details-62cc1e4"}}}
```

Create unprivileged `abitur-deploy` with no Docker group membership. Sudoers:

```sudoers
abitur-deploy ALL=(root) NOPASSWD: /usr/local/sbin/abitur-deploy backend, /usr/local/sbin/abitur-deploy web
```

Use separate per-repository SSH keys and forced commands in authorized_keys:
`restrict,command="/usr/local/sbin/abitur-ci-ssh backend" ssh-ed25519 ...`
and the corresponding `web` command for the other key. Disable forwarding and
PTY (the OpenSSH `restrict` option does both). The wrapper only accepts exact
SSH_ORIGINAL_COMMAND `deploy`. CI submits JSON stdin with exactly `image`, `sha`,
`registry_username`, `registry_token`; credentials never enter process arguments.
Set each repo's secrets `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_PRIVATE_KEY`,
`DEPLOY_KNOWN_HOSTS` (port 22, pinned host key, StrictHostKeyChecking=yes).
No long-lived registry PAT or package visibility change is required.

The root helper ignores SSH hangup/termination after accepting input, so Actions
cancellation cannot interrupt an in-flight server transaction. Do not kill it
manually during deployment. Registry credentials live in a temporary 0700 Docker
config and are removed on normal completion/failure. Commands and HTTP operations
have bounded timeouts; no command output or raw exception is forwarded to CI.

Pull and revision validation precede pin replacement. Backend explicitly runs
permissions then migrations before starting only api/bot with no dependencies,
no build and pull never. Web changes only web. Health checks use local public
profile/login and stable bot container identity/restart count. No paid model or
Telegram request is made. Failure restores only the component's old pin and
restarts its services; database schema is **not downgraded**. A rollback can itself
fail (for example an incompatible migration); CI then fails and root must inspect.
The other component pin, environment, resources, volumes and settings stay intact.
`/opt/abitur/deploy-receipts.json` records current/previous digest and SHA per
component after health passes. Retain those images and initial local rollback
images; never blanket prune, `down -v`, or downgrade the database for rollback.

## Retry and manual application rollback

Retry the failed Actions run only while its checked SHA is still main. An old run
will intentionally skip; push/merge a new checked commit to deliver a newer main.
To roll back an application image manually, root must acquire the same lock and
use `previous_image` for only the affected component from the receipt. Preserve
the other component entries. For example, with `component` set literally to
`backend` or `web`, a root-operated Python script can use this helper's
`atomic_json`, `PINS`, `RECEIPT`, `LOCK`, `SERVICES`, `Host` and
`start(host, component, migrate=False)` under `fcntl.flock(..., LOCK_EX)`.
Choose an already retained local image and leave migrations disabled; this does
not undo schema changes. Verify the public endpoints and bot state, then record
the operator intervention in the deployment receipt. Never edit pins concurrently
with Actions or force an old commit through the automatic main freshness guard.

Exact healthy retries are no-ops: migrations are not repeated and the rollback
receipt is preserved. An unhealthy same-image retry repairs the application and
retains the original previous image. Valid GitHub bot actors (`[bot]`) are accepted.
Failure diagnostics expose only fixed stage, exception class, numeric subprocess
return code and validated digest/SHA; never raw stderr or credential data. The
same safe record is written to `/opt/abitur/deploy-failure.json`, leaving the last
good receipt intact. `images.yaml` is canonical: editing old minimal overlay pins
alone cannot roll back a component, because the third overlay overrides them.
