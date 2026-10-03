# EPOCH — Multi-Tenant Cognitive Corpus

EPOCH ingests the recorded cognitive traces of domain experts (audio narration,
annotated documents, screen recordings) and turns them into queryable "cognitive
twins" — agents that can be asked questions and respond grounded in that expert's
actual reasoning.

EPOCH is **multi-tenant**: multiple hospitals (or organizations) share the same
platform, each with their own experts and traces. The core engineering problem
this project solves is **data isolation** — proving, structurally, that Hospital
B can never see Hospital A's data, even if the agent hallucinates a bad query or
a client sends a malicious header.

This repo is built incrementally across a 3-session capstone. This document
covers the whole project, what Session 15.1 delivers, and what's coming next.

---

## Project structure

```
.
├── .venv/              # Python virtual environment (already provisioned)
├── epoch_ingest.py     # Session 15.1 deliverable — multi-tenant isolation layer
└── README.md           # this file
```

---

## Setup — cloning and running this repo

### Prerequisites
- Python 3.12+ (this project was built and tested on 3.12.3)
- A [Google AI Studio](https://aistudio.google.com/) API key (Gemini) — only
  required for the optional live-Gemini demo, not for the CI assertions

### 1. Clone the repo

```bash
git clone <your-repo-url> epoch
cd epoch
```

> This directory is not yet a git repository. If you're setting up version
> control for the first time, run `git init`, commit the files, and push to
> your remote of choice (GitHub, GitLab, etc.) before others can clone it.

### 2. Create and activate a virtual environment

A `.venv` is already present in this project. To recreate it elsewhere:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install google-generativeai numpy
```

### 4. (Optional) Set your Gemini API key

Only needed if you want to run the live-Gemini demo (Section 11's
`run_two_tenant_demo(..., use_gemini=True)`), not for the default script run.

```bash
export GEMINI_API_KEY=your_key_here
```

```python
import google.generativeai as genai
genai.configure(api_key=YOUR_GEMINI_API_KEY)
```

### 5. Run it

```bash
python epoch_ingest.py
```

No API key or network access is required for the default run — everything is
verified with synthetic embeddings and an in-memory SQLite database.

---

## What Session 15.1 builds

`epoch_ingest.py` implements four independent isolation layers ("Red Circles"),
each one closing off a different way tenant data could leak, plus a synthetic
two-hospital corpus and a security regression suite that proves the isolation
holds.

### The scenario
- **Hospital A (Acme Diagnostics)** — Dr. Sarah Chen, oncologist. Her audio
  narrations, PDF diagnoses, and PACS screen recordings are ingested as her
  cognitive twin's corpus.
- **Hospital B (Bravo Medical)** — a separate tenant with its own expert
  (Dr. Arjun Patel, cardiologist). Hospital B's doctors must never see
  Dr. Chen's traces or Hospital A's clinical data — not through a lucky query,
  not through a coincidentally-reused thread ID, not through a hallucinated
  unfiltered SQL query, not through a missing/malformed request header.

### Red Circle 1 — Vector isolation (`FakeFAISSRouter`)
Simulates one FAISS index file per `{hospital_id}/{expert_id}`. Only one
expert's index is ever loaded into memory at a time; loading a new one evicts
the old one. Hospital B's search can never touch Hospital A's vectors because
Hospital A's index file is never opened.

### Red Circle 2 — Memory isolation (`FakeMemorySaverCompositeKey`)
A LangGraph-style checkpointer keyed on a **composite** primary key
`(hospital_id, expert_id, thread_id, step)` instead of just `thread_id`. This
defeats thread-ID guessing attacks: two hospitals can reuse the same
`thread_id` value safely, because the composite key makes a cross-hospital
read a guaranteed miss, not a coincidence-dependent leak.

### Red Circle 3 — SQL isolation (`FakePostgresRLS`)
Simulates Postgres Row-Level Security on the trace metadata table. Even an
agent that hallucinates an unfiltered `SELECT * FROM expert_traces` is
transparently scoped to the current hospital by the (simulated) RLS policy.
Deny-by-default: if the tenant context is never set, zero rows are returned.

### Red Circle 4 — Gateway (`FakeFastAPIGateway`)
Simulates FastAPI middleware that is the **single** trust boundary where a raw,
untrusted `X-Hospital-ID` header becomes a typed, trusted value passed
downstream. Missing header → `401`. Header that fails validation (path
traversal, SQL injection, log injection via regex `^[a-z0-9_-]{3,64}$`) →
`400`.

### Synthetic corpus
- `CHEN_TRACES` — 5 traces for Dr. Chen (3 audio, 2 PDF, 1 screen recording)
- `PATEL_TRACES` — 2 traces for Dr. Patel (1 audio, 1 PDF)
- `seed_corpus()` loads both into the FAISS router and the RLS-protected table

### `answer_node()`
A LangGraph-style node that retrieves top-k traces scoped to
`{hospital_id}/{expert_id}`, sets the RLS tenant context, and calls Gemini
(`gemini-2.5-flash`) to answer grounded in only that expert's traces. It
**consumes** `hospital_id` from state — it never decides it. Authentication
happens exactly once, at the gateway.

### CI assertions (security regression tests)
Five assertions run on every execution of the script and must never be
deleted — they are the machine-readable form of the isolation guarantees:

1. `assert_no_cross_expert_retrieval` — Hospital B's FAISS search never
   returns a Hospital A trace, even with a semantically perfect probe.
2. `assert_no_checkpoint_bleed` — a reused `thread_id` across hospitals
   never collides; the legitimate hospital can still read its own checkpoint.
3. `assert_rls_enforced` — Hospital B's unfiltered query returns zero
   Hospital A rows.
4. `assert_missing_header_rejected` — missing `X-Hospital-ID` → `401`.
5. `assert_invalid_header_rejected` — path traversal / injection payloads in
   `X-Hospital-ID` → `400`.

### Deliberate bug exercises
Two exercises intentionally break an isolation layer to make the failure
visceral, then restore it:

- `exercise_break_composite_key` — shows what a `thread_id`-only WHERE clause
  leaks across hospitals.
- `exercise_rls_guc_unset` — shows that an unset tenant context returns zero
  rows (safe failure), not all rows (unsafe failure).

### End-to-end demo
`run_two_tenant_demo()` sends three probes through the gateway: a legitimate
Hospital A query, an attack probe from Hospital B targeting Dr. Chen's data,
and a request with a missing header — confirming no Hospital A trace IDs ever
appear in a Hospital B response.

---

## Expected output

```
╔══════════════════════════════════════════════════════╗
║  EPOCH — Session 15.1 Verification                   ║
╚══════════════════════════════════════════════════════╝

  Seeded Hospital A (Dr. Chen): 5 traces
  Seeded Hospital B (Dr. Patel): 2 traces

  Verification: 5/5 checks passed in XXms

  ✓ Hospital B cannot retrieve Hospital A traces
  ✓ Composite key blocks cross-hospital checkpoint read
  ✓ RLS blinds Hospital B to Hospital A rows
  ✓ Missing X-Hospital-ID → 401
  ✓ Path traversal / injection in X-Hospital-ID → 400

  ✓ Session 15.1 COMPLETE.

  [EXERCISE] Breaking the composite key...
  EXPECTED FAIL: Hospital B reads Hospital A data: {'secret': 'Hospital A data'}
  → This is what happens without the composite key.
  → The fix: restore hospital_id to the WHERE clause.

  [EXERCISE] RLS deny-by-default when GUC is unset...
  GUC unset → rows returned: 0 (expected: 0, deny-by-default)
  PASS: deny-by-default confirmed

END-TO-END DEMO — Two-Tenant Probe
============================================================

[1] Hospital A asks about Chen GGO protocol → status 200
    Answer: [hospital_a/chen] Based on ...

[2] Hospital B probes for Hospital A (Dr. Chen) data → status 200
    Answer: [hospital_b/chen] No relevant traces found.
    CONFIRMED: no Hospital A trace IDs in Hospital B response

[3] Missing X-Hospital-ID → status 401 ✓
```

If any check fails, the script exits with status code 1 and prints the
assertion error before it — nothing downstream (exercises, demo) runs.

---

## What's next: Session 15.2

Session 15.2 replaces the header-based tenant identifier with real
authentication and adds **role-based access control** on top of the same
isolation shape:

- `X-Hospital-ID` → a **signed JWT**. The gateway middleware shape stays
  identical (one verification point, same propagation path into
  `graph_invoke`) — only the trust mechanism changes, from "trust whatever
  string arrives" to "trust only a cryptographically verified claim."
- The JWT carries a `role` claim (e.g. `viewer`, `admin`) in addition to
  `hospital_id`.
- `FakeMemorySaverCompositeKey` gains a `role` column — the composite key is
  **extended**, not replaced.
- Tool access is bound per-request via `bind_tools()` based on role: an admin
  role can reach admin-only tools; a viewer role cannot.
- Two new CI assertions are added on top of the five from this session
  (never delete or weaken the existing five):
  1. Role bindings are correct (a JWT with `role=viewer` cannot invoke
     admin-only tools).
  2. Admin tools are structurally unreachable to a viewer — not just
     unlisted, but impossible to invoke even under adversarial prompting.

Everything built in Session 15.1 is a permanent foundation:
`hospital_a/chen` and `hospital_b/patel` FAISS indexes, the composite-key
checkpointer, and the gateway middleware all carry forward unchanged in
shape — Session 15.2 builds on top of them rather than replacing them.

Session 15.3 (beyond 15.2) is expected to layer in the remaining production
concerns not yet covered here — real FAISS/Postgres/GCS backends in place of
the in-memory simulators, and hardening the ingestion pipeline itself.
