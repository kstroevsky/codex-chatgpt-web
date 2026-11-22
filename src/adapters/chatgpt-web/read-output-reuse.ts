import { createHash } from "node:crypto";
import type { CodexContentPart, CodexToolResultMessage } from "../../types";
import { isNativeThreadReader, readNativeThreadSnapshot, type ExecutionReceiptLedger } from "./execution-receipts";
import type { BrokerToolRequest } from "./turn-broker";

const BLOCK_LINES = 6;
const MAX_RESULT_LINES = 2_000;
const MAX_VISIBLE_BLOCKS = 2_048;
const MAX_VISIBLE_IMAGES = 32;
const MAX_RESULT_OVERRIDES = 256;
const digest = (text: string): string => createHash("sha256").update(text).digest("hex");
const sourceRead = /(?:^|[;|&\n])\s*(?:cat\s|rg\s|sed\s+-n\s|nl\s+-ba\s)/;

interface PreparedReadResult {
  message: CodexToolResultMessage;
  kind?: "shell" | "screenshot" | "thread";
  commit(): void;
}

/** One physical Web turn. Native Codex keeps the original result for every call. */
export class ChatGptReadOutputReuse {
  private readonly seenBlocks = new Set<string>();
  private readonly seenImages = new Set<string>();
  private readonly priorThreads = new Map<string, string>();
  readonly overrides = new Map<string, CodexToolResultMessage>();

  constructor(checkpoint?: ExecutionReceiptLedger) {
    for (const item of checkpoint?.receipts ?? []) {
      if (item.threadSnapshotVersion) this.priorThreads.set(item.threadSnapshotVersion, item.excerpt);
    }
  }

  prepare(original: CodexToolResultMessage, request?: BrokerToolRequest): PreparedReadResult {
    let visible = original;
    let kind: PreparedReadResult["kind"];
    let remember: () => void = () => {};
    if (request && !request.freeform && !original.isError) {
      if (request.wireName === "view_image" || request.wireName === "functions.view_image") {
        const image = this.screenshots(original);
        visible = image.message;
        remember = () => this.remember(this.seenImages, image.delivered, MAX_VISIBLE_IMAGES);
        if (visible !== original) kind = "screenshot";
      } else if (isNativeThreadReader(request.wireName)) {
        visible = this.thread(original, request.arguments ?? {});
        if (visible !== original) kind = "thread";
      } else if ((request.wireName === "exec_command" || request.wireName === "functions.exec_command")
        && typeof request.arguments?.cmd === "string" && sourceRead.test(request.arguments.cmd)) {
        const shell = this.shell(original);
        visible = shell.message;
        remember = () => this.remember(this.seenBlocks, shell.delivered, MAX_VISIBLE_BLOCKS);
        if (visible !== original) kind = "shell";
      }
    }
    return {
      message: visible,
      ...(kind ? { kind } : {}),
      commit: () => {
        remember();
        if (visible !== original) {
          this.overrides.set(original.toolCallId, visible);
          if (this.overrides.size > MAX_RESULT_OVERRIDES) this.overrides.delete(this.overrides.keys().next().value!);
        }
      },
    };
  }

  private remember(target: Set<string>, values: string[], limit: number): void {
    for (const value of values) {
      target.delete(value);
      target.add(value);
      if (target.size > limit) target.delete(target.values().next().value!);
    }
  }

  private screenshots(original: CodexToolResultMessage) {
    const delivered: string[] = [];
    if (typeof original.content === "string") return { message: original, delivered };
    let reused = 0;
    const parts: CodexContentPart[] = [];
    for (const part of original.content) {
      if (part.type === "image" && part.imageUrl.startsWith("data:image/")) {
        const key = digest(JSON.stringify([part.imageUrl, part.detail]));
        if (this.seenImages.has(key)) {
          reused++;
          parts.push({ type: "text", text: "[Identical screenshot already supplied earlier in this response. Full result retained in Codex history.]" });
          continue;
        }
        delivered.push(key);
      }
      parts.push(part);
    }
    return { message: reused ? { ...original, content: parts } : original, delivered };
  }

  private thread(original: CodexToolResultMessage, args: Record<string, unknown>): CodexToolResultMessage {
    // Wide first-page reads are usually a context refresh. Leave explicit
    // output detail, narrow reads and cursor-based requests available as-is.
    if (typeof args.threadId !== "string" || typeof args.turnLimit !== "number" || args.turnLimit < 3
      || args.cursor !== undefined || args.includeTurns === false || args.includeOutputs === true
      || !this.priorThreads.size) return original;
    const tested = readNativeThreadSnapshot(original, args.threadId);
    const prior = tested?.version ? this.priorThreads.get(tested.version) : undefined;
    if (!prior || tested?.excerpt !== prior) return original;
    return { ...original, content: [
      "The requested thread page was checked again. Its saved revision, settled turns and checkpoint excerpt match the earlier read.",
      prior,
      "Codex retained the complete tool result. For a missing detail, request a narrow page or the specific outputs.",
    ].join("\n") };
  }

  private shell(original: CodexToolResultMessage) {
    const delivered: string[] = [];
    const content = original.content;
    if (typeof content !== "string") return { message: original, delivered };
    const boundary = /\r?\n(?:Output|Final output):\r?\n/.exec(content);
    if (!boundary) return { message: original, delivered };
    const start = boundary.index + boundary[0].length;
    const header = content.slice(0, start);
    if (!/^Process exited with code 0\r?$/m.test(header)) return { message: original, delivered };
    const lines = content.slice(start).split("\n");
    if (lines.length > MAX_RESULT_LINES) return { message: original, delivered };

    const windows: Array<string | undefined> = [];
    for (let i = 0; i + BLOCK_LINES <= lines.length; i++) {
      const block = lines.slice(i, i + BLOCK_LINES);
      if (block.some(line => /^(?:\s*---|Warning:|Total output lines:|\s*…)/.test(line))) continue;
      const text = block.join("\n");
      if (text.length >= 240) windows[i] = digest(text);
    }
    const visible: string[] = [];
    const removed = new Set<number>();
    for (let i = 0; i < lines.length; ){
      let repeatedLines = 0;
      while (windows[i + repeatedLines] && this.seenBlocks.has(windows[i + repeatedLines]!)) repeatedLines += BLOCK_LINES;
      if (!repeatedLines) { visible.push(lines[i++]!); continue; }
      visible.push(`[${repeatedLines} identical output lines already supplied in this response. Full result retained in Codex history.]`);
      for (let n = 0; n < repeatedLines; n++) removed.add(i + n);
      i += repeatedLines;
    }
    windows.forEach((key, i) => {
      if (key && !lines.slice(i, i + BLOCK_LINES).some((_, n) => removed.has(i + n))) delivered.push(key);
    });
    return { message: removed.size ? { ...original, content: header + visible.join("\n") } : original, delivered };
  }
}
