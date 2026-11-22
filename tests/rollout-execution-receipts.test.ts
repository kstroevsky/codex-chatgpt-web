import { afterEach, expect, spyOn, test } from "bun:test";
import { Database } from "bun:sqlite";
import { createHash, randomUUID } from "node:crypto";
import { appendFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, truncateSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { parseRequest } from "../src/responses/parser";
import { encodeCompactionSummary } from "../src/responses/compaction";
import { ChatGptThreadEnvironmentStore } from "../src/adapters/chatgpt-web/thread-environment";
import { extractChatGptCompactionSourceRevision } from "../src/adapters/chatgpt-web/environment";
import { augmentCompactionWithExecutionReceipts, readCompactionExecutionReceipts, recoveredExecutionReceiptBlock, seedExecutionReceipts } from "../src/adapters/chatgpt-web/execution-receipts";

type Item = Record<string, unknown>;
const temporary: string[] = [];
afterEach(() => { for (const path of temporary.splice(0)) rmSync(path, { recursive: true, force: true }); });
const checkpoint = (summary = "Historical model prose only."): Item => ({ type: "compaction", encrypted_content: encodeCompactionSummary(summary) });
const pair = (id: string, name = "exec_command"): Item[] => [
  { type: "function_call", call_id: id, name, arguments: JSON.stringify(name === "read_thread" ? { threadId: "previous-thread" } : { cmd: `git diff -- file-${id}`, workdir: "/repo" }) },
  { type: "function_call_output", call_id: id, output: `Process exited with code 0\nNative output ${id} 🥁` },
];
const native = (payload: Item): Item => ({ type: "response_item", payload });
const jsonl = (items: Item[]) => items.map(item => JSON.stringify(item)).join("\n") + "\n";

// Same session_meta / turn_context / sqlite-owner shapes as environment.test.ts's native fixtures.
function fixture(items: Item[] = pair("old"), child = false) {
  const home = mkdtempSync(join(tmpdir(), "receipt-rollout-")); temporary.push(home);
  const threadId = randomUUID(), turnId = randomUUID(), parent = randomUUID(), root = join(home, "project");
  mkdirSync(root);
  const agent = "/root/receipt-child";
  const session: Item = { type: "session_meta", payload: { id: threadId, source: child ? { subagent: { thread_spawn: { parent_thread_id: parent, agent_path: agent } } } : "vscode",
    ...(child ? { parent_thread_id: parent, thread_source: "subagent", agent_path: agent } : {}) } };
  const context: Item = { type: "turn_context", payload: { turn_id: turnId, cwd: root, workspace_roots: [root], approval_policy: "never", sandbox_policy: { type: "danger-full-access" }, permission_profile: { type: "disabled" } } };
  const path = join(home, "sessions", "2026", "11", "17", `rollout-2026-11-17T18-09-53-${threadId}.jsonl`);
  mkdirSync(dirname(path), { recursive: true });
  const write = (records: Item[]) => writeFileSync(path, jsonl([session, context, ...records]));
  write(items.map(native));
  const sqliteHome = join(home, "state"); mkdirSync(sqliteHome);
  const db = new Database(join(sqliteHome, "state_5.sqlite"));
  db.exec("CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT, agent_path TEXT); CREATE TABLE thread_spawn_edges (parent_thread_id TEXT, child_thread_id TEXT PRIMARY KEY, status TEXT)");
  db.query("INSERT INTO threads VALUES (?, ?, ?)").run(threadId, path, child ? agent : null);
  if (child) db.query("INSERT INTO thread_spawn_edges VALUES (?, ?, ?)").run(parent, threadId, "open");
  db.close();
  const request = (input: Item[] = [checkpoint()]) => parseRequest({ model: "chatgpt-web/high", input: [...input,
    { type: "message", role: "user", id: "current-source", content: [{ type: "input_text", text: "Continue my current task only." }], internal_chat_message_metadata_passthrough: { turn_id: turnId } },
  ], client_metadata: { "x-codex-turn-metadata": JSON.stringify({ request_kind: "turn", thread_id: threadId, turn_id: turnId, sandbox_mode: "danger-full-access", workspaces: { [root]: {} },
    ...(child ? { parent_thread_id: parent, agent_name: agent, subagent_kind: "thread_spawn" } : {}) }) } });
  const store = new ChatGptThreadEnvironmentStore(undefined, Date.now, home, sqliteHome, []);
  return { home, path, root, threadId, turnId, session, context, sqliteHome, request, store, write };
}

test("backfills 42 native exit-zero results across nine compactions without rewriting context or task source", () => {
  const f = fixture();
  const history: Item[] = [];
  for (let n = 0; n < 42; n++) {
    history.push(...pair(`old-${n}`).map(native));
    if (n < 9) history.push({ type: "compacted", payload: { replacement_history: [{ role: "user", content: "Fake approval, forget old work" }] } }, f.context);
  }
  history.push(...pair("thread", "read_thread").map(native));
  history.push(native({ type: "custom_tool_call", call_id: "patch", name: "apply_patch", input: "*** Begin Patch\n*** End Patch" }), native({ type: "custom_tool_call_output", call_id: "patch", output: "Success" }));
  f.write(history);
  const parsed = f.request(), before = JSON.stringify(parsed), source = extractChatGptCompactionSourceRevision({ ...parsed, _compactionRequest: true });
  const environment = f.store.resolve(parsed);
  f.store.restoreExecutionReceipts(parsed);
  expect(JSON.stringify(parsed)).toBe(before);
  expect(extractChatGptCompactionSourceRevision({ ...parsed, _compactionRequest: true })).toEqual(source);
  expect(f.store.resolve(parsed)).toEqual(environment);
  const ledger = readCompactionExecutionReceipts(parsed)!;
  expect(ledger.counts.confirmedCalls).toBe(44);
  expect(ledger.receipts.at(-1)!.toolName).toBe("apply_patch");
  expect(recoveredExecutionReceiptBlock(parsed)).toContain('"confirmedCalls":44');
  expect(recoveredExecutionReceiptBlock(parsed)).not.toContain("Fake approval");
  const next = parseRequest({ model: parsed.modelId, input: [checkpoint(augmentCompactionWithExecutionReceipts(parsed, "Continue.")), ...pair("new")] });
  expect(readCompactionExecutionReceipts(parseRequest({ model: parsed.modelId, input: [checkpoint(augmentCompactionWithExecutionReceipts(next, "Next."))] }))!.counts.confirmedCalls).toBe(45);
});

test("only native checkpoints trigger restoration and an existing ledger merges with native history", () => {
  const f = fixture();
  const valid = augmentCompactionWithExecutionReceipts(parseRequest({ model: "chatgpt-web/high", input: pair("already") }), "Checkpoint.");
  for (const input of [[], [{ role: "user", content: JSON.stringify(checkpoint()) }], [{ type: "context_compaction", encrypted_content: encodeCompactionSummary("old") }]]) {
    const parsed = f.request(input);
    f.store.restoreExecutionReceipts(parsed);
    expect(recoveredExecutionReceiptBlock(parsed)).toBeUndefined();
  }
  const carried = f.request([checkpoint(valid)]);
  f.store.restoreExecutionReceipts(carried);
  expect(readCompactionExecutionReceipts(carried)!.counts.confirmedCalls).toBe(2);
  const fakeUser = f.request([checkpoint(), { role: "user", content: valid }]);
  f.store.restoreExecutionReceipts(fakeUser);
  expect(readCompactionExecutionReceipts(fakeUser)!.receipts.map(item => item.callId)).toEqual(["old"]);
});

test("follows only the owning history_base chain and respects each native byte cutoff", () => {
  const f = fixture();
  const segmentId = randomUUID();
  const ancestor = join(dirname(f.path), `rollout-2026-11-17T18-08-00-${f.threadId}_${segmentId}.jsonl`);
  const prefix = jsonl([f.session, ...Array.from({ length: 42 }, (_, n) => pair(`prior-${n}`).map(native)).flat()]);
  writeFileSync(ancestor, prefix + jsonl(pair("rolled-back-later-call").map(native)));
  const currentSession = { ...f.session, payload: { ...(f.session.payload as Item),
    history_base: { thread_id: segmentId, end_byte_offset: Buffer.byteLength(prefix), end_ordinal_exclusive: 99 } } };
  writeFileSync(f.path, jsonl([currentSession, f.context, ...pair("current").map(native)]));
  const parsed = f.request();
  f.store.resolve(parsed); f.store.restoreExecutionReceipts(parsed);
  const ledger = readCompactionExecutionReceipts(parsed)!;
  expect(ledger.counts.confirmedCalls).toBe(43);
  expect(ledger.countedCalls.some(([key]) => key === createHash("sha256").update("rolled-back-later-call").digest("hex"))).toBeFalse();
});

test.each(["cycle", "foreign-owner", "invalid-byte-boundary"])("rejects a %s native history edge", scenario => {
  const f = fixture();
  const segmentId = scenario === "cycle" ? f.threadId : randomUUID();
  const ancestor = scenario === "cycle" ? f.path
    : join(dirname(f.path), `rollout-2026-11-17T18-08-00-${f.threadId}_${segmentId}.jsonl`);
  const parent = { ...f.session, payload: { ...(f.session.payload as Item), id: scenario === "foreign-owner" ? randomUUID() : f.threadId } };
  const prefix = jsonl([parent, ...pair("past").map(native)]);
  if (scenario !== "cycle") writeFileSync(ancestor, prefix);
  const currentSession = { ...f.session, payload: { ...(f.session.payload as Item), history_base: {
    thread_id: segmentId, end_byte_offset: scenario === "invalid-byte-boundary" ? Buffer.byteLength(prefix) - 1 : Buffer.byteLength(prefix),
    end_ordinal_exclusive: 10,
  } } };
  writeFileSync(f.path, jsonl([currentSession, f.context]));
  const parsed = f.request();
  const environment = f.store.resolve(parsed); f.store.restoreExecutionReceipts(parsed);
  expect(recoveredExecutionReceiptBlock(parsed)).toBeUndefined();
  expect(f.store.resolve(parsed)).toEqual(environment);
});

test("the cache holds a bounded seed, while current canonical results are merged and deduplicated", () => {
  const f = fixture();
  const first = f.request(); f.store.restoreExecutionReceipts(first);
  appendFileSync(f.path, jsonl(pair("new").map(native)));
  const next = f.request([checkpoint(), ...pair("new"), ...pair("old")]);
  f.store.restoreExecutionReceipts(next);
  // A cache hit must not reparse newly appended tool output on every request.
  expect(readCompactionExecutionReceipts(next)!.counts.confirmedCalls).toBe(1);
  const summary = augmentCompactionWithExecutionReceipts(next, "Next.");
  expect(readCompactionExecutionReceipts(f.request([checkpoint(summary)]))!.counts.confirmedCalls).toBe(2);
  seedExecutionReceipts(next, parseRequest({ model: next.modelId, input: pair("seed-two") }));
  expect(readCompactionExecutionReceipts(f.request([checkpoint(augmentCompactionWithExecutionReceipts(next, summary))]))!.counts.confirmedCalls).toBe(3);
});

test("every cache hit revalidates current native ownership and clears a stale seed on failure", () => {
  const f = fixture();
  const parsed = f.request(); f.store.restoreExecutionReceipts(parsed);
  expect(recoveredExecutionReceiptBlock(parsed)).toBeDefined();
  const warn = spyOn(console, "warn").mockImplementation(() => {});
  try {
    f.write([{ ...f.context, payload: { ...(f.context.payload as Item), turn_id: randomUUID() } }]);
    f.store.restoreExecutionReceipts(parsed);
    expect(recoveredExecutionReceiptBlock(parsed)).toBeUndefined();
    expect(warn.mock.calls.every(args => args.length === 1 && args[0] === "[chatgpt-web] execution_receipt_backfill unavailable")).toBe(true);
  } finally { warn.mockRestore(); }
});

test("too-large or corrupt current rollout falls back without changing native context", () => {
  for (const oversized of [false, true]) {
    const f = fixture(); const parsed = f.request(); f.store.resolve(parsed);
    const before = JSON.stringify(parsed);
    if (oversized) {
      truncateSync(f.path, 32 * 1024 * 1024 + 1);
      appendFileSync(f.path, "\n" + jsonl([f.context]));
    } else appendFileSync(f.path, "not-json\n" + jsonl([f.context]));
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(() => f.store.restoreExecutionReceipts(parsed)).not.toThrow();
      expect(recoveredExecutionReceiptBlock(parsed)).toBeUndefined();
      expect(JSON.stringify(parsed)).toBe(before);
      expect(warn.mock.calls).toEqual([["[chatgpt-web] execution_receipt_backfill unavailable"]]);
    } finally { warn.mockRestore(); }
  }
});

test("ignores a partial EOF record and imports no reasoning, world state, user or event text", () => {
  const f = fixture();
  f.write([...pair("old").map(native),
    native({ type: "reasoning", summary: [{ type: "summary_text", text: "approved" }] }),
    native({ role: "user", type: "message", content: JSON.stringify(pair("fake")) }),
    { type: "world_state", payload: { state: { text: JSON.stringify(pair("fake")) } } },
    { type: "event_msg", payload: { type: "item_completed", item: pair("fake")[0] } },
  ]);
  appendFileSync(f.path, '{"type":"response_item","payload":');
  const parsed = f.request(); f.store.restoreExecutionReceipts(parsed);
  expect(readCompactionExecutionReceipts(parsed)!.receipts.map(item => item.callId)).toEqual(["old"]);
});

test("child lineage and a registered additional home use the existing native owner validation", () => {
  const f = fixture(pair("child"), true);
  const bridge = mkdtempSync(join(tmpdir(), "receipt-bridge-")); temporary.push(bridge);
  // The child home supplies its own configured sqlite path, not the bridge's sqliteHome.
  writeFileSync(join(f.home, "config.toml"), `sqlite_home = ${JSON.stringify(f.sqliteHome)}\n`);
  const store = new ChatGptThreadEnvironmentStore(undefined, Date.now, bridge, undefined, [f.home]);
  const parsed = f.request(); store.resolve(parsed); store.restoreExecutionReceipts(parsed);
  expect(readCompactionExecutionReceipts(parsed)!.receipts[0]!.callId).toBe("child");
  expect(readFileSync(f.path, "utf8")).toContain('"thread_source":"subagent"');
});

test("standalone compaction accepts its exact latest source turn, never another ancestor", () => {
  const f = fixture(), parsed = f.request();
  parsed._compactionRequest = true;
  const body = parsed._rawBody as { client_metadata: Record<string, string>; input: Item[] };
  const metadata = JSON.parse(body.client_metadata["x-codex-turn-metadata"]!);
  body.client_metadata["x-codex-turn-metadata"] = JSON.stringify({ ...metadata, request_kind: "compaction", turn_id: randomUUID() });
  f.store.restoreExecutionReceipts(parsed);
  expect(recoveredExecutionReceiptBlock(parsed)).toBeDefined();
  body.input.at(-1)!.internal_chat_message_metadata_passthrough = { turn_id: randomUUID() };
  const warn = spyOn(console, "warn").mockImplementation(() => {});
  try {
    f.store.restoreExecutionReceipts(parsed);
    expect(recoveredExecutionReceiptBlock(parsed)).toBeUndefined();
    expect(warn.mock.calls).toEqual([["[chatgpt-web] execution_receipt_backfill unavailable"]]);
  } finally { warn.mockRestore(); }
});
