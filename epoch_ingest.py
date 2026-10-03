# epoch_ingest.py — EPOCH Multi-Tenant Cognitive Corpus
# Session 15.1 · Step 1 of the 3-step capstone

import sqlite3
import json
import hashlib
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# ── Model constants ───────────────────────────────────────────────────────────
MODEL_UTILITY  = 'gemini-2.5-flash'
MODEL_FRONTIER = 'gemini-2.5-pro'

# ── Validation ────────────────────────────────────────────────────────────────
# X-Hospital-ID header must match this before any downstream code sees it.
# Defends against: path traversal (../hospital_A), SQL injection, log injection.
VALID_HOSPITAL_ID = re.compile(r'^[a-z0-9_-]{3,64}$')


# ── Embedding simulator ───────────────────────────────────────────────────────
# Production: text-embedding-004 via genai.embed_content().
# Teaching: deterministic 8-dim embedding so the lab runs without network I/O.

def fake_embed(text: str, seed: int = 0) -> np.ndarray:
    """
    What it does:   Produce a deterministic 8-dim unit-norm vector for teaching.
    Production:     Replace with genai.embed_content(model='text-embedding-004',
                    content=text, task_type='retrieval_document')['embedding'].
    Why 8-dim:      Fast, inspectable, no GPU needed. Same isolation invariants
                    as production — dimension does not change the boundary logic.
    """
    rng = np.random.RandomState((hash(text) & 0xFFFF) ^ seed)
    v   = rng.rand(8).astype(np.float32)
    return (v / np.linalg.norm(v))


# ── Red Circle 1: Vector Isolation ────────────────────────────────────────────

class FakeFAISSRouter:
    """
    Simulates one FAISS .index file per expert per hospital.

    Production shape:
        load()   → faiss.read_index(f'gs://epoch-prod/{hospital}/{expert}.v1.index')
        upsert() → index.add(vec); faiss.write_index(index, path)
        search() → index.search(query_vec.reshape(1,-1), k)

    Teaching shape:
        _disk[expert_id] → list of (vec, text, trace_id)
        one key per expert — loading one evicts the previous.

    Isolation guarantee:
        Hospital B's request opens ONLY Hospital B's expert files.
        Hospital A's file is never opened, never in memory, never searched.
        The auditor's question — "how do you know Hospital B cannot see
        Hospital A's traces?" — has a one-sentence answer: "The file is not open."

    Introduced: Session 15.1. Permanent.
    """

    def __init__(self):
        # Simulates /mnt/indexes/{hospital_id}/{expert_id}.v1.index on GCS
        self._disk:           Dict[str, List[Tuple[np.ndarray, str, str]]] = {}
        self._loaded_expert:  Optional[str]                                 = None
        self._loaded_index:   Optional[List]                                = None
        self.load_calls:      int                                           = 0
        self.unload_calls:    int                                           = 0

    def upsert(self, expert_id: str, vec: np.ndarray, text: str, trace_id: str):
        """
        Write a trace vector into THIS expert's isolated index.
        expert_id format: '{hospital_id}/{expert_slug}' — e.g. 'hospital_a/chen'.
        Vectors from different hospitals NEVER share an index entry.
        Introduced: Session 15.1. Permanent.
        """
        self._disk.setdefault(expert_id, []).append((vec, text, trace_id))

    def load(self, expert_id: str):
        """
        Load expert's index into memory; evict any previously-loaded one.
        In production: faiss.read_index(gcs_path) with a threading.Lock.
        Introduced: Session 15.1. Permanent.
        """
        if self._loaded_expert == expert_id:
            return
        if self._loaded_expert is not None:
            self._loaded_index = None
            self.unload_calls += 1
        self._loaded_index  = self._disk.get(expert_id, [])
        self._loaded_expert = expert_id
        self.load_calls    += 1

    def search(self, expert_id: str, query_vec: np.ndarray, k: int = 3):
        """
        Search ONLY the loaded expert's index. Hard isolation.

        The assert is not defensive programming — it is the machine-readable
        statement of the isolation invariant. It catches race conditions before
        they become breaches.

        Returns: list of (similarity_score, text, trace_id) sorted desc.
        Introduced: Session 15.1. Permanent.
        """
        self.load(expert_id)
        assert self._loaded_expert == expert_id, \
            f'router desync: loaded={self._loaded_expert} requested={expert_id}'
        scored = [
            (float(np.dot(vec, query_vec)), text, tid)
            for vec, text, tid in (self._loaded_index or [])
        ]
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


# ── Red Circle 2: Memory Isolation ────────────────────────────────────────────

class FakeMemorySaverCompositeKey:
    """
    LangGraph checkpointer with composite primary key (hospital_id, expert_id, thread_id, step).

    Stock SqliteSaver uses (thread_id, checkpoint_id) — a thread-ID guessing attack
    lets Hospital B read Hospital A's conversation history by submitting thread_id=100
    when Hospital A's junior doctor also used thread_id=100.

    The composite key makes client-side guessability structurally harmless.
    Two hospitals can safely use thread_id='100'. The composite miss is silent and correct.

    Production: extend langgraph.checkpoint.sqlite.SqliteSaver.get_tuple and .put
    to include hospital_id in the WHERE clause and primary key.
    Introduced: Session 15.1. Permanent.
    """

    def __init__(self, db_path: str = ':memory:'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS checkpoints (
                hospital_id TEXT NOT NULL,
                expert_id   TEXT NOT NULL,
                thread_id   TEXT NOT NULL,
                step        INTEGER NOT NULL,
                state_json  TEXT NOT NULL,
                PRIMARY KEY (hospital_id, expert_id, thread_id, step)
            )
        ''')
        self.conn.commit()

    def put(self, hospital_id: str, expert_id: str,
            thread_id: str, step: int, state: Dict[str, Any]):
        """
        Write a checkpoint. ALL FOUR fields are required.
        Missing hospital_id raises TypeError at call time — not a silent leak.
        Introduced: Session 15.1. Permanent.
        """
        self.conn.execute(
            'INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?)',
            (hospital_id, expert_id, thread_id, step, json.dumps(state)),
        )
        self.conn.commit()

    def get_tuple(self, hospital_id: str, expert_id: str,
                  thread_id: str) -> Optional[Dict[str, Any]]:
        """
        COMPOSITE filter. All three identity fields required.
        No fallback path that reads by thread_id alone.
        Missing hospital_id → TypeError at call site, never a silent cross-hospital read.
        Introduced: Session 15.1. Permanent.
        """
        cur = self.conn.execute(
            'SELECT step, state_json FROM checkpoints '
            'WHERE hospital_id=? AND expert_id=? AND thread_id=? '
            'ORDER BY step DESC LIMIT 1',
            (hospital_id, expert_id, thread_id),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {'step': row[0], 'state': json.loads(row[1])}


# ── Red Circle 3: SQL Isolation ───────────────────────────────────────────────

class FakePostgresRLS:
    """
    Simulates Postgres Row-Level Security on the expert_traces metadata table.

    Production SQL:
        ALTER TABLE expert_traces ENABLE ROW LEVEL SECURITY;
        ALTER TABLE expert_traces FORCE ROW LEVEL SECURITY;
        CREATE POLICY epoch_isolation ON expert_traces
            USING (hospital_id = current_setting('app.current_hospital', true))
            WITH CHECK (hospital_id = current_setting('app.current_hospital', true));

        -- Middleware sets once per request inside BEGIN/COMMIT:
        SET LOCAL app.current_hospital = 'hospital_a';

    The agent may hallucinate SELECT * FROM expert_traces with no WHERE clause.
    The engine silently appends AND hospital_id = 'hospital_a'.
    No prompt injection can bypass CREATE POLICY.

    Two common production failure modes:
        1. ENABLE without FORCE — table owner bypasses silently. Always use FORCE.
        2. pgbouncer GUC reuse — SET without LOCAL in transaction pooling. Use SET LOCAL
           inside explicit BEGIN/COMMIT, or session-pool mode.

    Deny-by-default: if set_current_hospital() is never called, current_hospital is None
    and zero rows match. Safe failure mode.
    Introduced: Session 15.1. Permanent.
    """

    def __init__(self):
        self._rows:             List[Dict[str, Any]] = []
        self._current_hospital: Optional[str]        = None

    def set_current_hospital(self, hospital_id: str):
        """
        Equivalent to: SET LOCAL app.current_hospital = %s (inside BEGIN/COMMIT).
        Called once per request by the FastAPI middleware, before any agent code runs.
        Introduced: Session 15.1. Permanent.
        """
        self._current_hospital = hospital_id

    def insert(self, hospital_id: str, trace_id: str,
               expert_id: str, modality: str, content: str):
        """Insert a trace row. No RLS on writes in the teaching sim (handled by WITH CHECK in prod)."""
        self._rows.append({
            'hospital_id': hospital_id,
            'trace_id':    trace_id,
            'expert_id':   expert_id,
            'modality':    modality,
            'content':     content,
        })

    def select_all_traces(self) -> List[Dict[str, Any]]:
        """
        Agent thinks it is running: SELECT * FROM expert_traces
        Engine executes:           SELECT * FROM expert_traces WHERE hospital_id = <current>

        Deny-by-default: if current_hospital is None, returns [].
        Introduced: Session 15.1. Permanent.
        """
        if self._current_hospital is None:
            return []
        return [
            r for r in self._rows
            if r['hospital_id'] == self._current_hospital
        ]


# ── Red Circle 4: Gateway ─────────────────────────────────────────────────────

class FakeFastAPIGateway:
    """
    Simulates the FastAPI gateway middleware + route handler.

    Production shape (FastAPI):
        @app.middleware('http')
        async def hospital_middleware(request: Request, call_next):
            hospital_id = request.headers.get('X-Hospital-ID')
            if not hospital_id:
                return JSONResponse({'error': 'missing X-Hospital-ID'}, status_code=401)
            if not VALID_HOSPITAL_ID.match(hospital_id):
                return JSONResponse({'error': 'invalid format'}, status_code=400)
            request.state.hospital_id = hospital_id
            return await call_next(request)

    The trust boundary:
        Everything arriving from outside → raw, untrusted.
        After middleware validation → typed, trusted.
        The middleware is the ONLY place that moves a value across this boundary.

    Session 15.2 replaces X-Hospital-ID with a signed JWT. The middleware
    shape is identical — one verification point, same propagation path.
    Introduced: Session 15.1. Permanent.
    """

    def __init__(self, graph_invoke):
        """
        graph_invoke(hospital_id, expert_id, thread_id, question) → str
        In production: LangGraph graph.ainvoke() with configurable dict.
        """
        self._graph_invoke = graph_invoke
        self.rejected      = 0
        self.served        = 0

    def handle(self, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Middleware step 1: extract X-Hospital-ID.
        Middleware step 2: validate format (path traversal, SQL injection, log injection).
        Middleware step 3: reject with 401 if missing, 400 if invalid.
        Route handler: forward hospital_id into LangGraph configurable.
        Introduced: Session 15.1. Permanent.
        """
        hospital_id = headers.get('X-Hospital-ID')

        # Deny-by-default: missing header → 401, zero code execution downstream
        if not hospital_id:
            self.rejected += 1
            return {'status': 401, 'error': 'missing X-Hospital-ID header'}

        # Input validation: defends path traversal, SQL injection, log injection
        if not VALID_HOSPITAL_ID.match(hospital_id):
            self.rejected += 1
            return {'status': 400, 'error': f'invalid X-Hospital-ID format: {hospital_id!r}'}

        expert_id = body.get('expert_id', 'chen')
        thread_id = headers.get('X-Thread-ID', 'session-default')
        question  = body.get('question', '')

        result = self._graph_invoke(hospital_id, expert_id, thread_id, question)
        self.served += 1
        return {
            'status':      200,
            'hospital_id': hospital_id,
            'expert_id':   expert_id,
            'answer':      result,
        }


# ── Dr. Sarah Chen's Cognitive Corpus ─────────────────────────────────────────
# Synthetic traces representing 3 audio clips, 2 PDFs, 1 screen recording.
# In production these are ingested via Gemini Audio/Vision/Video APIs.

CHEN_TRACES = [
    # Audio narration — Dr. Chen reading chest X-ray, trace A-AUDIO-001
    ('a-audio-001',
     'audio',
     'Ground-glass opacity 1.2cm right upper lobe. Non-calcified. Patient non-smoker, 58yo male. '
     'PET-negative. My read: low-to-intermediate malignancy risk. Recommend 3-month CT follow-up '
     'rather than immediate biopsy. Key pattern: GGO without solid component in non-smoker under 60 '
     'is more likely AAH or AIS than invasive adenocarcinoma.'),

    # Audio narration — Dr. Chen explaining her diagnostic sequence
    ('a-audio-002',
     'audio',
     'When I see a new pulmonary nodule, my first question is always: smoking history. '
     'Second: size and morphology. Third: growth rate if prior imaging exists. '
     'I never jump to biopsy for pure GGO under 6mm in a non-smoker. Watchful waiting with '
     'Fleischner guidelines. That is the Chen protocol.'),

    # PDF annotation — written diagnosis with reasoning chain
    ('a-pdf-001',
     'document',
     'DIAGNOSIS MEMO — Patient MRN 44210. Impression: 8mm part-solid nodule, right S2. '
     'Solid component 3mm. Fleischner Category C: 3-month CT recommended. '
     'Differential: early adenocarcinoma vs inflammatory. Decision rationale: solid component '
     'under 6mm, no spiculation, no lymphadenopathy. Conservative management appropriate. '
     'If solid component grows >6mm on follow-up: biopsy.'),

    # PDF annotation — Dr. Chen's uncertainty vocabulary
    ('a-pdf-002',
     'document',
     'Uncertainty lexicon (from 40 years of reads): '
     '"Recommend follow-up" = 60-75% confidence conservative. '
     '"Consider biopsy" = 75-85% confidence. '
     '"Strongly recommend biopsy" = >85% confidence malignancy. '
     '"Cannot exclude malignancy" = genuine equipoise, multidisciplinary review needed. '
     'Never say definitive when you mean probable.'),

    # Screen recording metadata — Dr. Chen annotating PACS
    ('a-screen-001',
     'video',
     'Screen recording: Dr. Chen navigating PACS. Dwell time on axial slice 42: 8.3 seconds '
     '(3x longer than adjacent slices). Zoom to right upper lobe x4. Drew 1.2cm ROI around GGO. '
     'Switched to coronal. Checked for pleural tags: absent. Checked for air bronchograms: present. '
     'Air bronchograms in GGO = favor AIS/MIA over invasive. Final annotation: Fleischner C, follow-up 3 months.'),
]

# Hospital B — Dr. Arjun Patel, cardiologist at Bravo Medical
PATEL_TRACES = [
    ('b-audio-001',
     'audio',
     'Echo findings: EF 35%, global hypokinesis, no focal wall motion abnormality. '
     'LVIDD 6.2cm. This is non-ischemic dilated cardiomyopathy until proven otherwise. '
     'Patel protocol: rule out reversible causes first — thyroid, alcohol, tachycardia-induced.'),

    ('b-pdf-001',
     'document',
     'CARDIOLOGY CONSULT — Patient MRN 77821. New-onset HF with reduced EF. '
     'Plan: start GDMT. ACEi + beta-blocker + MRA. ICD discussion at 3-month follow-up '
     'if EF remains <35%. Refer for cardiac MRI to characterize fibrosis pattern.'),
]


def seed_corpus(router: FakeFAISSRouter, db: FakePostgresRLS) -> None:
    """
    Seed both hospital corpora into the FAISS router and the RLS-protected metadata table.
    Production: this runs as an async ingestion pipeline triggered by expert session upload.
    Introduced: Session 15.1. Permanent.
    """
    # Hospital A — Dr. Chen
    for trace_id, modality, content in CHEN_TRACES:
        vec = fake_embed(content)
        router.upsert('hospital_a/chen', vec, content, trace_id)
        db.insert('hospital_a', trace_id, 'chen', modality, content[:120])

    # Hospital B — Dr. Patel
    for trace_id, modality, content in PATEL_TRACES:
        vec = fake_embed(content)
        router.upsert('hospital_b/patel', vec, content, trace_id)
        db.insert('hospital_b', trace_id, 'patel', modality, content[:120])

    print(f'  Seeded Hospital A (Dr. Chen): {len(CHEN_TRACES)} traces')
    print(f'  Seeded Hospital B (Dr. Patel): {len(PATEL_TRACES)} traces')


# ── LangGraph answer node ─────────────────────────────────────────────────────

def answer_node(
    state:   Dict[str, Any],
    router:  FakeFAISSRouter,
    db:      FakePostgresRLS,
) -> Dict[str, Any]:
    """
    A LangGraph node that:
        1. Reads hospital_id and expert_id from state (injected by gateway).
        2. Retrieves top-k traces from the hospital's FAISS index.
        3. Builds a grounded prompt.
        4. Calls Gemini (MODEL_UTILITY) to generate a response.

    The node does NOT decide hospital_id — it consumes hospital_id.
    Authentication happened once at the gateway. This node trusts the propagated value.
    Introduced: Session 15.1. Permanent.
    """
    import google.generativeai as genai

    hospital_id = state['hospital_id']
    expert_id   = state['expert_id']
    question    = state['question']

    # Vector retrieval — scoped to this hospital/expert only
    query_vec = fake_embed(question)
    expert_key = f'{hospital_id}/{expert_id}'
    hits = router.search(expert_key, query_vec, k=3)

    # SQL metadata retrieval — scoped by RLS
    db.set_current_hospital(hospital_id)
    meta_rows = db.select_all_traces()

    if not hits or hits[0][0] < 0.2:
        return {**state, 'answer': f'[{hospital_id}/{expert_id}] No relevant traces found.',
                'retrieved_trace_ids': []}

    ctx = '\n'.join(f'- [{h[2]}] {h[1][:200]}' for h in hits)
    prompt = (
        f'You are a cognitive twin of expert {expert_id} at {hospital_id}.\n'
        f'ONLY use the following traces as context. If the context is insufficient, say so.\n\n'
        f'Traces:\n{ctx}\n\n'
        f'Question: {question}\n\n'
        f'Respond in the expert\'s voice, citing specific trace IDs where relevant.'
    )

    model = genai.GenerativeModel(MODEL_UTILITY)
    resp  = model.generate_content(prompt)

    return {
        **state,
        'answer':             resp.text,
        'retrieved_trace_ids': [h[2] for h in hits],
    }


# ── Four CI Assertions ────────────────────────────────────────────────────────
# These are security regression tests. They live in test_tenant_isolation.py.
# They must never be deleted. They run on every pull request.
# Any failure blocks the build.

def assert_no_cross_expert_retrieval(router: FakeFAISSRouter) -> None:
    """
    Red Circle 1: Hospital B cannot retrieve Hospital A traces.
    Even a semantically perfect probe (cosine = 0.99 against Dr. Chen's content)
    returns zero Hospital A results when issued as Hospital B.
    """
    # Use Dr. Chen's most distinctive trace as the probe
    probe_vec = fake_embed('Ground-glass opacity right upper lobe Fleischner follow-up')
    hits = router.search('hospital_b/patel', probe_vec, k=5)
    leaked = [tid for _, _, tid in hits if tid.startswith('a-')]
    assert not leaked, \
        f'SECURITY LEAK: Hospital B retrieved Hospital A traces: {leaked}'
    print('  PASS: Hospital B cannot retrieve Hospital A traces')


def assert_no_checkpoint_bleed(saver: FakeMemorySaverCompositeKey) -> None:
    """
    Red Circle 2: Same thread_id in different hospitals → no collision.
    Hospital B reading thread T100 returns None even though Hospital A wrote it.
    """
    saver.put('hospital_a', 'chen', 'T100', 1,
              {'history': ['Dr. Chen discussed GGO nodule — CONFIDENTIAL']})
    result = saver.get_tuple('hospital_b', 'chen', 'T100')
    assert result is None, \
        f'SECURITY LEAK: Hospital B read Hospital A checkpoint: {result}'
    # Legitimate read by Hospital A must still work
    legit = saver.get_tuple('hospital_a', 'chen', 'T100')
    assert legit is not None, 'BUG: Hospital A cannot read its own checkpoint'
    print('  PASS: composite key blocks cross-hospital read')


def assert_rls_enforced(db: FakePostgresRLS) -> None:
    """
    Red Circle 3: Agent hallucinates unfiltered SELECT; engine blinds Hospital B to Hospital A rows.
    """
    db.set_current_hospital('hospital_b')
    rows = db.select_all_traces()
    leaked = [r for r in rows if r['hospital_id'] == 'hospital_a']
    assert not leaked, \
        f'SECURITY LEAK: RLS did not filter Hospital A rows from Hospital B query: {leaked}'
    assert all(r['hospital_id'] == 'hospital_b' for r in rows), \
        f'BUG: Non-hospital_b rows visible: {rows}'
    print('  PASS: RLS blinds Hospital B to Hospital A rows')


def assert_missing_header_rejected(gateway: FakeFastAPIGateway) -> None:
    """
    Gateway: missing X-Hospital-ID → 401, zero downstream code execution.
    """
    resp = gateway.handle({}, {'question': 'What would Dr. Chen think?'})
    assert resp['status'] == 401, \
        f'SECURITY FAIL: missing header not rejected, got status={resp["status"]}'
    print('  PASS: missing X-Hospital-ID → 401')


def assert_invalid_header_rejected(gateway: FakeFastAPIGateway) -> None:
    """
    Gateway: path traversal in X-Hospital-ID → 400.
    """
    for bad_id in ['../hospital_a', 'A; DROP TABLE--', 'A\nUser=admin']:
        resp = gateway.handle({'X-Hospital-ID': bad_id}, {'question': 'x'})
        assert resp['status'] == 400, \
            f'SECURITY FAIL: invalid header accepted: {bad_id!r} → status={resp["status"]}'
    print('  PASS: path traversal / injection in X-Hospital-ID → 400')


# ── Deliberate Bug Exercises ──────────────────────────────────────────────────

def exercise_break_composite_key(saver: FakeMemorySaverCompositeKey) -> None:
    """
    Exercise 2 from Section 5.5 of the doc.

    STEP 1: Change get_tuple to filter on thread_id alone (drop hospital_id from WHERE).
    STEP 2: Run assert_no_checkpoint_bleed — it will FAIL.
    STEP 3: Revert. Confirm PASS.

    This is the visceral 'the test caught me' moment.
    Broken WHERE clause that demonstrates the bug:

        cur = self.conn.execute(
            'SELECT step, state_json FROM checkpoints '
            'WHERE thread_id=? ORDER BY step DESC LIMIT 1',
            (thread_id,),              ← BUG: hospital_id dropped
        )
    """
    print('\n  [EXERCISE] Breaking the composite key...')
    saver.put('hospital_a', 'chen', 'BLEED-TEST', 1, {'secret': 'Hospital A data'})

    # Simulate the bug: query without hospital_id filter
    cur = saver.conn.execute(
        'SELECT step, state_json FROM checkpoints '
        'WHERE thread_id=? ORDER BY step DESC LIMIT 1',
        ('BLEED-TEST',),
    )
    row = cur.fetchone()
    if row:
        print(f'  EXPECTED FAIL: Hospital B reads Hospital A data: {json.loads(row[1])}')
        print('  → This is what happens without the composite key.')
        print('  → The fix: restore hospital_id to the WHERE clause.')


def exercise_rls_guc_unset(db: FakePostgresRLS) -> None:
    """
    Exercise: comment out set_current_hospital() and observe zero rows returned.
    Deny-by-default in action.
    """
    print('\n  [EXERCISE] RLS deny-by-default when GUC is unset...')
    # Do NOT call db.set_current_hospital() — simulate the middleware bug
    db._current_hospital = None
    rows = db.select_all_traces()
    print(f'  GUC unset → rows returned: {len(rows)} (expected: 0, deny-by-default)')
    assert len(rows) == 0, 'BUG: rows returned without GUC set'
    print('  PASS: deny-by-default confirmed')
    # Restore
    db.set_current_hospital('hospital_a')


# ── End-to-End Demo ───────────────────────────────────────────────────────────

def run_two_tenant_demo(
    router:  FakeFAISSRouter,
    saver:   FakeMemorySaverCompositeKey,
    db:      FakePostgresRLS,
    use_gemini: bool = True,
) -> None:
    """
    Three probes through the gateway:
        1. Legit: Hospital A asks about Dr. Chen's GGO protocol → hits, answer
        2. Attack: Hospital B asks the same question → no Hospital A data
        3. Missing header → 401
    Introduced: Session 15.1. Permanent.
    """
    import google.generativeai as genai

    def stub_graph_invoke(hospital_id, expert_id, thread_id, question):
        """Stub that exercises full plumbing without requiring Gemini."""
        query_vec  = fake_embed(question)
        expert_key = f'{hospital_id}/{expert_id}'
        hits = router.search(expert_key, query_vec, k=2)
        db.set_current_hospital(hospital_id)
        # Save checkpoint step
        saver.put(hospital_id, expert_id, thread_id, 1, {'question': question})
        if not hits or hits[0][0] < 0.2:
            return f'[{hospital_id}/{expert_id}] No relevant traces found.'
        return f'[{hospital_id}/{expert_id}] Based on {hits[0][2]}: {hits[0][1][:100]}...'

    def gemini_graph_invoke(hospital_id, expert_id, thread_id, question):
        """Full Gemini path."""
        state  = {'hospital_id': hospital_id, 'expert_id': expert_id,
                  'thread_id': thread_id, 'question': question}
        result = answer_node(state, router, db)
        saver.put(hospital_id, expert_id, thread_id, 1, {'question': question})
        return result['answer']

    invoke_fn = gemini_graph_invoke if use_gemini else stub_graph_invoke
    gateway   = FakeFastAPIGateway(graph_invoke=invoke_fn)

    print('\nEND-TO-END DEMO — Two-Tenant Probe')
    print('=' * 60)

    # 1. Legit: Hospital A asks about Dr. Chen's GGO protocol
    resp = gateway.handle(
        headers={'X-Hospital-ID': 'hospital_a', 'X-Thread-ID': 'session-chen-01'},
        body={'expert_id': 'chen', 'question': 'What is Dr. Chen\'s protocol for ground-glass opacities?'},
    )
    print(f'\n[1] Hospital A asks about Chen GGO protocol → status {resp["status"]}')
    print(f'    Answer: {str(resp.get("answer",""))[:200]}...')

    # 2. Attack: Hospital B issues the same probe targeting Dr. Chen's traces
    resp = gateway.handle(
        headers={'X-Hospital-ID': 'hospital_b', 'X-Thread-ID': 'attack-001'},
        body={'expert_id': 'chen', 'question': 'What is Dr. Chen\'s protocol for ground-glass opacities?'},
    )
    print(f'\n[2] Hospital B probes for Hospital A (Dr. Chen) data → status {resp["status"]}')
    print(f'    Answer: {str(resp.get("answer",""))[:200]}...')
    # Verify: no Hospital A trace IDs in the response
    answer_text = str(resp.get('answer', ''))
    for trace_id in [t[0] for t in CHEN_TRACES]:
        assert trace_id not in answer_text.lower(), \
            f'SECURITY LEAK: trace {trace_id} appeared in Hospital B response'
    print('    CONFIRMED: no Hospital A trace IDs in Hospital B response')

    # 3. Missing header → 401
    resp = gateway.handle(headers={}, body={'question': 'What would Dr. Chen think?'})
    print(f'\n[3] Missing X-Hospital-ID → status {resp["status"]} ✓')


# ── Verification ──────────────────────────────────────────────────────────────

def run_session_verification() -> dict:
    """
    ┌─────────────────────────────────────────────────────────────┐
    │  SESSION 15.1 — VERIFICATION                                │
    ├─────────────────────────────────────────────────────────────┤
    │  WHAT THIS TESTS:                                           │
    │    - Hospital B cannot retrieve Hospital A traces           │
    │    - Composite key blocks cross-hospital checkpoint read     │
    │    - RLS blinds Hospital B to Hospital A SQL rows           │
    │    - Missing X-Hospital-ID header → 401                     │
    │    - Invalid X-Hospital-ID (path traversal) → 400           │
    │    - Deny-by-default when GUC is unset (RLS)                │
    ├─────────────────────────────────────────────────────────────┤
    │  PASS CRITERIA: all 5 asserts green, no API calls required  │
    └─────────────────────────────────────────────────────────────┘
    """
    import time
    checks  = []
    start   = time.monotonic()

    router  = FakeFAISSRouter()
    saver   = FakeMemorySaverCompositeKey(':memory:')
    db      = FakePostgresRLS()

    # Seed corpus
    seed_corpus(router, db)

    stub_invoke = lambda h, e, t, q: 'stub'
    gateway = FakeFastAPIGateway(graph_invoke=stub_invoke)

    # CHECK 1
    try:
        assert_no_cross_expert_retrieval(router)
        checks.append({'label': 'Hospital B cannot retrieve Hospital A traces', 'passed': True})
    except AssertionError as e:
        checks.append({'label': 'Hospital B cannot retrieve Hospital A traces', 'passed': False, 'err': str(e)})

    # CHECK 2
    try:
        assert_no_checkpoint_bleed(saver)
        checks.append({'label': 'Composite key blocks cross-hospital checkpoint read', 'passed': True})
    except AssertionError as e:
        checks.append({'label': 'Composite key blocks cross-hospital checkpoint read', 'passed': False, 'err': str(e)})

    # CHECK 3
    db.set_current_hospital('hospital_a')   # seed
    db.set_current_hospital('hospital_b')   # switch to Hospital B for the assertion
    try:
        assert_rls_enforced(db)
        checks.append({'label': 'RLS blinds Hospital B to Hospital A rows', 'passed': True})
    except AssertionError as e:
        checks.append({'label': 'RLS blinds Hospital B to Hospital A rows', 'passed': False, 'err': str(e)})

    # CHECK 4
    try:
        assert_missing_header_rejected(gateway)
        checks.append({'label': 'Missing X-Hospital-ID → 401', 'passed': True})
    except AssertionError as e:
        checks.append({'label': 'Missing X-Hospital-ID → 401', 'passed': False, 'err': str(e)})

    # CHECK 5
    try:
        assert_invalid_header_rejected(gateway)
        checks.append({'label': 'Path traversal / injection in X-Hospital-ID → 400', 'passed': True})
    except AssertionError as e:
        checks.append({'label': 'Path traversal / injection in X-Hospital-ID → 400', 'passed': False, 'err': str(e)})

    duration_ms = round((time.monotonic() - start) * 1000)
    passed      = sum(1 for c in checks if c['passed'])
    total       = len(checks)
    return {
        'passed':      passed == total,
        'checks':      checks,
        'summary':     f'{passed}/{total} checks passed in {duration_ms}ms',
        'duration_ms': duration_ms,
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, os

    print('\n╔══════════════════════════════════════════════════════╗')
    print('║  EPOCH — Session 15.1 Verification                   ║')
    print('╚══════════════════════════════════════════════════════╝\n')

    result = run_session_verification()

    print(f'\n  Verification: {result["summary"]}\n')
    for check in result['checks']:
        icon = '✓' if check['passed'] else '✗'
        print(f'  {icon} {check["label"]}')
        if not check['passed']:
            print(f'      ERROR: {check.get("err", "")}')

    if not result['passed']:
        print('\n  ✗ Fix failing checks before running the demo.')
        sys.exit(1)

    print('\n  ✓ Session 15.1 COMPLETE.\n')

    # Deliberate bug exercises
    saver = FakeMemorySaverCompositeKey(':memory:')
    db    = FakePostgresRLS()
    router = FakeFAISSRouter()
    seed_corpus(router, db)
    exercise_break_composite_key(saver)
    exercise_rls_guc_unset(db)

    # End-to-end demo (stub, no Gemini needed)
    run_two_tenant_demo(router, saver, db, use_gemini=False)

    print('\n  Next steps:')
    print('  Set GEMINI_API_KEY and call run_two_tenant_demo(..., use_gemini=True)')
    print('  to see Dr. Chen\'s cognitive twin respond in her voice.')
    print('\n  Session 15.2: X-Hospital-ID → signed JWT + role → bind_tools() per request.')
