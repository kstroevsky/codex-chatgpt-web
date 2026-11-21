import {
  CHATGPT_WEB_PLATFORM_RESERVE_TOKENS, chatGptWebImageTokenReserve,
} from "../../chatgpt-web-models";
import { estimateTokens } from "../../lib/token-estimate";
import type { CodexToolResultMessage, CodexUsage } from "../../types";
import { estimateCompiledChatGptWebInputTokens } from "./input-tokens";
import type { CompiledChatGptWebPrompt } from "./prompt";
import type { BrokerToolResult } from "./turn-broker";

interface BrowserUsageSnapshot {
  accountKey: string;
  tokens: number;
}

type BrowserUsageSource = "retained" | "fresh_resume";

/**
 * In-memory estimate of one browser submission and its actual continuation.
 * A finished receipt can seed the next turn only on a verified reused lease.
 * Preparing an upload alone does not establish context on the model.
 */
export class ChatGptRetainedUsage {
  private selected?: { accountKey?: string; reused: boolean; prompt?: { tokens: number; excerpted: boolean } };
  private accepted?: { accountKey?: string; source?: BrowserUsageSource; inputTokens: number };
  private finished?: BrowserUsageSnapshot;
  private deliveredToolTokens = 0;
  private previousOutputTokens = 0;
  private readonly deliveredCalls = new Set<string>();
  private readonly rounds = new Map<string, CodexUsage>();

  constructor(private readonly modelId: string, private readonly prior?: BrowserUsageSnapshot) {}

  select(reused: boolean, accountKey?: string): void {
    // A rejected physical attempt contributes no context to a fresh retry.
    this.selected = { reused, accountKey };
    this.accepted = undefined;
    this.finished = undefined;
    this.deliveredCalls.clear();
    this.rounds.clear();
    this.deliveredToolTokens = 0;
    this.previousOutputTokens = 0;
  }

  prepare(prompt: CompiledChatGptWebPrompt, transactionId?: string): void {
    if (this.selected) {
      this.selected.prompt = {
        tokens: estimateCompiledChatGptWebInputTokens(prompt, this.modelId, transactionId),
        excerpted: Boolean(prompt.historicalShellExcerpts),
      };
    }
  }

  accept(): void {
    const selected = this.selected;
    if (!selected?.prompt || this.accepted) return;
    const prior = selected.reused ? this.prior : undefined;
    if (selected.reused && (!prior || prior.accountKey !== selected.accountKey)) return;
    const source = selected.reused ? "retained" : selected.prompt.excerpted ? "fresh_resume" : undefined;
    // Every rendered prompt estimate includes the platform reserve. An epoch
    // carries it once, even if it has multiple Continue submissions.
    this.accepted = {
      accountKey: selected.accountKey,
      source,
      inputTokens: selected.prompt.tokens + (prior ? prior.tokens - CHATGPT_WEB_PLATFORM_RESERVE_TOKENS : 0),
    };
  }

  get source(): BrowserUsageSource | undefined {
    return this.accepted?.source;
  }

  addToolResult(message: CodexToolResultMessage, delivered: BrokerToolResult): void {
    if (!this.accepted || this.deliveredCalls.has(message.toolCallId)) return;
    // Images are charged by their image reserve. The literal base64 is wire
    // encoding and is not text in the model context.
    const content = delivered.content.map(part => {
      if (!part || typeof part !== "object" || (part as { type?: unknown }).type !== "image") return part;
      const { data: _data, ...image } = part as Record<string, unknown>;
      return image;
    });
    const imageTokens = typeof message.content === "string" ? 0 : message.content.reduce((total, part) =>
      total + (part.type === "image" && part.imageUrl.startsWith("data:") ? chatGptWebImageTokenReserve(part.detail) : 0), 0);
    this.deliveredToolTokens += imageTokens + estimateTokens(JSON.stringify({
      role: "tool", name: message.toolName, ...delivered, content,
    }), this.modelId);
    this.deliveredCalls.add(message.toolCallId);
  }

  /** Journal each physical model round once, including re-emitted tool batches. */
  recordRound(toolCallIds: readonly string[], outputTokens: number, final: boolean): CodexUsage | undefined {
    const accepted = this.accepted;
    if (!accepted || (!final && toolCallIds.length === 0)) return undefined;
    const key = final ? "final" : JSON.stringify([...toolCallIds].sort());
    const priorRound = this.rounds.get(key);
    if (priorRound) return accepted.source ? priorRound : undefined;
    const inputTokens = accepted.inputTokens + this.deliveredToolTokens + this.previousOutputTokens;
    const usage = { inputTokens, outputTokens, totalTokens: inputTokens + outputTokens, estimated: true };
    this.previousOutputTokens += outputTokens;
    this.rounds.set(key, usage);
    if (final && accepted.accountKey) this.finished = { accountKey: accepted.accountKey, tokens: usage.totalTokens };
    // Complete fresh sends keep the capacity forecast. Retained turns and accepted
    // fresh replays with excerpts price their actual browser history instead.
    return accepted.source ? usage : undefined;
  }

  snapshot(): BrowserUsageSnapshot | undefined {
    return this.finished;
  }
}
