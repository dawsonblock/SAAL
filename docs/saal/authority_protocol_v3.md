# Authority Protocol v3: authenticated identity and cumulative limits

## Trust model

The acting agent sends an untrusted one-line JSON request to a Unix socket.
The authority obtains its caller UID from the **kernel socket peer credentials**
(`SO_PEERCRED` on Linux, `getpeereid` on supported BSD/macOS runtimes), then
checks that UID against an administrator-installed list of `(principal, task_id)`
identities. A claimed principal alone cannot select a different installed
delegation. Unsupported platforms fail closed.

The peer mapping, delegation file, authorization HMAC and human credential are
separate private inputs under the authority UID; each key/policy is opened through
an `O_NOFOLLOW` descriptor and checked with `fstat`. Avoid same-UID deployment
where the actor could read these inputs. UID binding is **not** user
authentication and cannot distinguish actors sharing the same UID.

## Accepted request

```json
{"action":{"tool":"schedule.propose","args":{"slot_id":"A","cost_cents":0},"principal":"demo_user","task_id":"demo-1"}}
```

Reject duplicate keys, malformed schemas, unknown actions, mismatched task/UID,
expired or revoked delegations, insufficient approval, cost and action quotas.
Requests over 16 KiB fail. External side effects are NOT executed by this
server; a trusted executor must separately consume the exact ticket with the
**same authoritative token database** before an effect. `SimulatedExecutor` is
for tests only.

## Capability claims

Version 3 adds `action_limit` bound by HMAC to the existing identity, action
hash, issued/expiry times, nonce, cost and task limit, and policy digest. The
SQLite registry tracks all spent tickets, plus outstanding unexpired tickets.
Both a cost reservation and a count reservation occur before returning ALLOW.
Consumption is single use; revocation affects outstanding tickets. Expired,
unconsumed reservations are released, while executed/spent tickets remain
counted. The authority service is the only intended owner of the registry.

## Remaining risks

- UID binding is insufficient if the untrusted process also controls the UID
  mapping or can send requests through a confused-deputy proxy.
- POSIX `0660` socket mode authorizes group access, not individual processes.
  Access control also requires a correctly protected parent directory.
- The toy executor shares an `Authority` instance and is **not** a deployed
  process boundary; no real calendar/financial/device adapter is included.
- No per-principal rate-limiter, anti-DoS queue, mTLS, service authentication,
  real-world effect re-verification or bounded network egress is implemented.
