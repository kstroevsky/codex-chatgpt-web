import { constants, closeSync, fstatSync, openSync, readSync, lstatSync } from "node:fs";
import { parseRequest } from "../../responses/parser";
import type { CodexParsedRequest } from "../../types";
import { resolveCurrentCodexRolloutEnvironment, resolveCodexRolloutAncestor } from "./codex-rollout-environment";
import { readCompactionExecutionReceipts, seedExecutionReceipts } from "./execution-receipts";

const MAX_ROLLOUT_BYTES = 32 * 1024 * 1024;
const MAX_BASELINES = 128;
const RESULT_TYPES = new Set(["function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output"]);
// Compact parsed containers retain only their bounded WeakMap ledger, never full tool outputs.
const baselines = new Map<string, { parsed: CodexParsedRequest; size: number; dev: number; ino: number; ancestors: Array<{ path: string; ino: number; mtime: number }> }>();
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value);

/** Native ownership is revalidated even on cache hits. This helper never supplies permissions. */
export function restoreRolloutExecutionReceipts(
  parsed: CodexParsedRequest,
  options: Parameters<typeof resolveCurrentCodexRolloutEnvironment>[0],
): boolean {
  const rollout = resolveCurrentCodexRolloutEnvironment(options);
  if (!rollout) return false;
  const key = JSON.stringify([options.lineage.threadId, options.turnId, rollout.rolloutPath]);
  const fd = openSync(rollout.rolloutPath, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile() || !Number.isSafeInteger(stat.size) || stat.size <= 0 || stat.size > MAX_ROLLOUT_BYTES) {
      throw new Error("Unavailable bounded native rollout");
    }
    const cached = baselines.get(key);
    if (cached && cached.dev === stat.dev && cached.ino === stat.ino && cached.size <= stat.size
      && cached.ancestors.every(item => {
        const metadata = lstatSync(item.path);
        return !metadata.isSymbolicLink() && metadata.ino === item.ino && metadata.mtimeMs === item.mtime;
      })) {
      baselines.delete(key); baselines.set(key, cached);
      seedExecutionReceipts(parsed, cached.parsed);
      return readCompactionExecutionReceipts(parsed) !== undefined;
    }
    const input: Record<string, unknown>[] = [];
    const ancestors: Array<{ path: string; ino: number; mtime: number }> = [];
    const visited = new Set<string>();
    let remaining = MAX_ROLLOUT_BYTES;
    const collect = (path: string, limit: number | undefined, current: boolean, depth: number): void => {
      if (depth > 16 || visited.has(path)) throw new Error("Cyclic or excessive native history");
      visited.add(path);
      const source = openSync(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
      try {
        const metadata = fstatSync(source);
        const size = limit ?? metadata.size;
        if (!metadata.isFile() || !Number.isSafeInteger(size) || size <= 0 || size > metadata.size || size > remaining) {
          throw new Error("Unavailable bounded native history segment");
        }
        remaining -= size;
        const bytes = Buffer.alloc(size);
        let offset = 0;
        while (offset < bytes.length) {
          const count = readSync(source, bytes, offset, Math.min(64 * 1024, bytes.length - offset), offset);
          if (!count) throw new Error("Native history changed during receipt recovery");
          offset += count;
        }
        if (limit !== undefined && bytes.at(-1) !== 0x0a) throw new Error("Invalid native history byte boundary");
        const complete = bytes.subarray(0, bytes.lastIndexOf(0x0a) + 1);
        const lines = new TextDecoder("utf-8", { fatal: true }).decode(complete).split("\n");
        const segment: Record<string, unknown>[] = [];
        let first = true;
        let latestTurn: unknown;
        let historyBase: Record<string, unknown> | undefined;
        for (const line of lines) {
          if (!line.trim()) continue;
          const item: unknown = JSON.parse(line.replace(/^\uFEFF/, ""));
          if (!record(item)) throw new Error("Invalid native history record");
          const payload = item.payload;
          if (first) {
            if (item.type !== "session_meta" || !record(payload) || payload.id !== options.lineage.threadId) {
              throw new Error("Native history owner changed during receipt recovery");
            }
            historyBase = record(payload.history_base) ? payload.history_base : undefined;
            first = false;
          }
          if (item.type === "turn_context" && record(payload)) latestTurn = payload.turn_id;
          if (item.type === "response_item" && record(payload) && typeof payload.type === "string" && RESULT_TYPES.has(payload.type)) segment.push(payload);
        }
        if (first || current && latestTurn !== options.turnId && (options.compactionSourceTurnId === undefined || latestTurn !== options.compactionSourceTurnId)) {
          throw new Error("Native rollout turn changed during receipt recovery");
        }
        if (historyBase) {
          if (typeof historyBase.thread_id !== "string" || !Number.isSafeInteger(historyBase.end_byte_offset)) {
            throw new Error("Invalid native history edge");
          }
          const ancestor = resolveCodexRolloutAncestor(rollout.rolloutPath,
            [options.codexHome, ...(options.additionalCodexHomes ?? [])], options.lineage, historyBase.thread_id);
          collect(ancestor, historyBase.end_byte_offset as number, false, depth + 1);
        }
        input.push(...segment);
        if (!current) ancestors.push({ path, ino: metadata.ino, mtime: metadata.mtimeMs });
      } finally { closeSync(source); }
    };
    collect(rollout.rolloutPath, undefined, true, 0);
    const history = parseRequest({ model: parsed.modelId, input });
    const baseline = parseRequest({ model: parsed.modelId, input: [] });
    seedExecutionReceipts(baseline, history);
    baselines.delete(key);
    baselines.set(key, { parsed: baseline, size: stat.size, dev: stat.dev, ino: stat.ino, ancestors });
    while (baselines.size > MAX_BASELINES) baselines.delete(baselines.keys().next().value!);
    seedExecutionReceipts(parsed, baseline);
    return readCompactionExecutionReceipts(parsed) !== undefined;
  } catch (error) {
    baselines.delete(key);
    throw error;
  } finally { closeSync(fd); }
}
