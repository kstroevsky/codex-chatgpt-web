import { expect, test } from "bun:test";
import { augmentCompactionWithExecutionReceipts, readCompactionExecutionReceipts } from "../src/adapters/chatgpt-web/execution-receipts";
import { ChatGptReadOutputReuse } from "../src/adapters/chatgpt-web/read-output-reuse";
import type { BrokerToolRequest } from "../src/adapters/chatgpt-web/turn-broker";
import { estimateChatGptWebUsage } from "../src/adapters/chatgpt-web/usage";
import { encodeCompactionSummary } from "../src/responses/compaction";
import { parseRequest } from "../src/responses/parser";
import type { CodexToolResultMessage } from "../src/types";

const caps = { localToolsEnabled: true, solAvailable: true, extraHighAvailable: false, proAvailable: false };
const request = (wireName: string, arguments_: Record<string, unknown> = {}): BrokerToolRequest => ({
  callId: "fixture", wireName, freeform: false, arguments: arguments_,
});
const result = (id: string, toolName: string, content: CodexToolResultMessage["content"]): CodexToolResultMessage => ({
  role: "toolResult", toolCallId: id, toolName, content, isError: false, timestamp: 0,
});
const shell = (id: string, start: number, end: number, changed?: number) => result(id, "exec_command",
  "Chunk ID: output\nWall time: 0.1000 seconds\nProcess exited with code 0\nOutput:\n"
  + Array.from({ length: end - start + 1 }, (_, index) => {
    const line = start + index;
    return `${line}\tconst setting${line} = "${line === changed ? "Changed value." : "Evidence for this source line."}";`;
  }).join("\n"));

test("only previously delivered identical file ranges are excerpted and all changed lines survive", () => {
  const reads = new ChatGptReadOutputReuse();
  const first = shell("first", 1, 22);
  const second = shell("next", 9, 30, 18);
  const original = structuredClone(second);
  const cmd = request("exec_command", { cmd: "nl -ba src/fixture.ts | sed -n '9,30p'" });
  const earlier = reads.prepare(first, cmd);
  expect(earlier.message).toBe(first);
  expect(reads.prepare(second, cmd).message).toBe(second);
  earlier.commit();

  const projected = reads.prepare(second, cmd);
  expect(projected.kind).toBe("shell");
  expect(projected.message.content).toContain("identical output lines");
  expect(projected.message.content).toContain("setting18");
  expect(projected.message.content).toContain("Changed value.");
  expect(projected.message.content).toContain("setting30");
  expect(projected.message.content).not.toContain("setting10");
  expect(second).toEqual(original);
  projected.commit();
  expect(reads.overrides.get(second.toolCallId)).toBe(projected.message);
  expect(new ChatGptReadOutputReuse().prepare(second, cmd).message).toBe(second);
});

test("identical screenshots are reused in one browser turn and usage counts the actual delivered data", () => {
  const reads = new ChatGptReadOutputReuse();
  const imageURL = "data:image/png;base64,Zmlyc3Q=";
  const first = result("first", "view_image", [{ type: "image", imageUrl: imageURL, detail: "high" }]);
  const next = result("next", "view_image", [
    { type: "image", imageUrl: imageURL, detail: "high" },
    { type: "image", imageUrl: imageURL, detail: "original" },
    { type: "text", text: "New caption captured by the tool" },
  ]);
  reads.prepare(first, request("view_image")).commit();
  const projected = reads.prepare(next, request("view_image"));
  expect(projected.kind).toBe("screenshot");
  expect(projected.message.content).toEqual([
    expect.objectContaining({ type: "text", text: expect.stringContaining("Identical screenshot") }),
    expect.objectContaining({ type: "image", detail: "original" }),
    expect.objectContaining({ type: "text", text: "New caption captured by the tool" }),
  ]);
  projected.commit();
  const p = parseRequest({ model: "gpt-5.6-sol", input: "Continue" });
  p.context.messages.push(first, next);
  const before = estimateChatGptWebUsage(p, { answer: "Continuing" }, caps, false);
  const after = estimateChatGptWebUsage(p, { answer: "Continuing" }, caps, false, false, undefined, undefined, reads.overrides);
  expect(before.inputTokens - after.inputTokens).toBeGreaterThan(4_000);
  expect(p.context.messages.at(-1)).toBe(next);
});

const threadPage = (updatedAt: number, turnId = "checked-turn", note = "Completed outbox decision.") => JSON.stringify({
  schemaVersion: 1, thread: { id: "saved-thread", hostId: "local", updatedAt }, page: { order: "newest_first" },
  turns: [{ id: turnId, status: "completed", items: [
    { type: "agentMessage", phase: "final_answer", text: note },
    { type: "mcpToolCall", output: "older tool detail ".repeat(1_800) },
  ] }],
});

test("a wide read of the same settled chat page reuses its checkpoint note and detailed or changed reads stay whole", () => {
  const wireName = "mcp__codex_app__read_thread";
  const parsed = parseRequest({ model: "gpt-5.6-sol", input: [
    { type: "function_call", name: "read_thread", namespace: "mcp__codex_app", call_id: "previous",
      arguments: JSON.stringify({ threadId: "saved-thread", turnLimit: 10 }) },
    { type: "function_call_output", call_id: "previous", output: threadPage(100) },
    { type: "function_call", name: "read_thread", namespace: "mcp__codex_app", call_id: "other-page",
      arguments: JSON.stringify({ threadId: "saved-thread", turnLimit: 8 }) },
    { type: "function_call_output", call_id: "other-page", output: threadPage(100, "other-page", "Other page evidence.") },
  ] });
  const checkpoint = encodeCompactionSummary(augmentCompactionWithExecutionReceipts(parsed, "Continue the task."));
  const input = parseRequest({ model: "gpt-5.6-sol", input: [{ type: "compaction", encrypted_content: checkpoint }] });
  const reads = new ChatGptReadOutputReuse(readCompactionExecutionReceipts(input));
  const args = { threadId: "saved-thread", turnLimit: 8 };
  const same = result("same", "read_thread", threadPage(100));
  const reused = reads.prepare(same, request(wireName, args));
  expect(reused.kind).toBe("thread");
  expect(reused.message.content).toContain("Completed outbox decision.");
  expect(reused.message.content).not.toContain("older tool detail");
  expect(reads.prepare(same, request(wireName, { ...args, turnLimit: 1 })).message).toBe(same);
  expect(reads.prepare(same, request(wireName, { ...args, includeOutputs: true })).message).toBe(same);
  expect(reads.prepare(same, request(wireName, { ...args, cursor: "next" })).message).toBe(same);
  for (const page of [threadPage(101), threadPage(100, "different-turn"), threadPage(100, "checked-turn", "Changed plan.")]) {
    const changed = result("changed", "read_thread", page);
    expect(reads.prepare(changed, request(wireName, args)).message).toBe(changed);
  }
  expect(same.content).toBe(threadPage(100));
});
