import { expect, test } from "bun:test";
import { CHATGPT_WEB_PLATFORM_RESERVE_TOKENS } from "../src/chatgpt-web-models";
import { estimateCompiledChatGptWebInputTokens } from "../src/adapters/chatgpt-web/input-tokens";
import { ChatGptRetainedUsage } from "../src/adapters/chatgpt-web/retained-usage";
import type { CompiledChatGptWebPrompt } from "../src/adapters/chatgpt-web/prompt";

const model = "gpt-5.6-sol";
const account = "a".repeat(64);
const tx = "ctx_" + "1".repeat(32);
const prompt = (text: string): CompiledChatGptWebPrompt => ({ text, images: [] });

test("retained usage carries real multipart overhead, one reserve, and actual results across Continue", () => {
  const original = new ChatGptRetainedUsage(model);
  const staged = { ...prompt("final instructions"),
    multipart: { parts: ['{"records":["earlier evidence"]}', '{"records":["last evidence"]}'], commit: "final instructions" } };
  original.select(false, account);
  original.prepare(staged, tx);
  original.accept();
  const sent = estimateCompiledChatGptWebInputTokens(staged, model, tx);
  expect(sent).toBeGreaterThan(CHATGPT_WEB_PLATFORM_RESERVE_TOKENS + 100);
  expect(original.recordRound([], 120, true)).toBeUndefined();
  expect(original.snapshot()?.tokens).toBe(sent + 120);

  const retained = new ChatGptRetainedUsage(model, original.snapshot());
  const delta = prompt("Continue with the next check.");
  retained.select(true, account);
  retained.prepare(delta, tx);
  expect(retained.recordRound(["check"], 80, false)).toBeUndefined();
  retained.accept();
  const batch = retained.recordRound(["check"], 80, false)!;
  expect(batch.inputTokens).toBe(sent + 120 + estimateCompiledChatGptWebInputTokens(delta, model, tx) - CHATGPT_WEB_PLATFORM_RESERVE_TOKENS);
  expect(batch.totalTokens).toBe(batch.inputTokens + 80);

  const visible = { role: "toolResult" as const, toolCallId: "check", toolName: "view_image", isError: false, timestamp: 0,
    content: [{ type: "image" as const, imageUrl: "data:image/png;base64,AAAA" }] };
  const wire = { content: [{ type: "image", mimeType: "image/png", data: "A".repeat(250_000) }] };
  retained.addToolResult(visible, wire);
  retained.addToolResult(visible, wire);
  expect(retained.recordRound(["check"], 80, false)).toEqual(batch);
  const final = retained.recordRound([], 25, true)!;
  const toolTokens = final.inputTokens - (batch.inputTokens + batch.outputTokens);
  expect(toolTokens).toBeGreaterThanOrEqual(4_096);
  expect(toolTokens).toBeLessThan(4_200);
  expect(retained.snapshot()?.tokens).toBe(final.totalTokens);
});

test("a different, unidentified, or unproven account cannot borrow a previous epoch estimate", () => {
  const prior = { accountKey: account, tokens: 89_000 };
  for (const current of ["b".repeat(64), undefined]) {
    const u = new ChatGptRetainedUsage(model, prior);
    u.select(true, current);
    u.prepare(prompt("small suffix"));
    u.accept();
    expect(u.recordRound([], 20, true)).toBeUndefined();
    expect(u.snapshot()).toBeUndefined();
  }
  const missing = new ChatGptRetainedUsage(model);
  missing.select(true, account);
  missing.prepare(prompt("small suffix"));
  missing.accept();
  expect(missing.recordRound([], 20, true)).toBeUndefined();
  expect(missing.snapshot()).toBeUndefined();
});

test("fresh retry resets the accepted retained attempt before recording another receipt", () => {
  const u = new ChatGptRetainedUsage(model, { accountKey: account, tokens: 89_000 });
  u.select(true, account);
  u.prepare(prompt("original attempted suffix"));
  u.accept();
  expect(u.recordRound(["one"], 20, false)?.inputTokens).toBeGreaterThan(89_000);
  u.select(false, account);
  const fresh = prompt("fresh replay");
  u.prepare(fresh);
  u.accept();
  expect(u.recordRound([], 10, true)).toBeUndefined();
  expect(u.snapshot()?.tokens).toBe(estimateCompiledChatGptWebInputTokens(fresh, model) + 10);
});
