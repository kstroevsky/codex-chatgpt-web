import { namespacedToolName, type CodexMessage, type CodexToolResultMessage } from "../../types";
import type { BrokerToolRequest } from "./turn-broker";

function toolKey(name: string, args: Record<string, unknown>): string | undefined {
  // An empty terminal poll observes current process state. Its old output cannot tell
  // whether the process has finished in the meantime.
  if (/^(?:functions[._]+)?write_stdin$/.test(name) && !args.chars) return undefined;
  return JSON.stringify([name, args], (_key, value: unknown) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) return value;
    const record = value as Record<string, unknown>;
    return Object.fromEntries(Object.keys(record).sort().map(key => [key, record[key]]));
  });
}

export interface ChatGptStallRecovery {
  completed: Map<string, CodexToolResultMessage>;
  partialAnswer: string;
}

/** Recover only from results joined to a call the stalled browser actually dispatched. */
export function createStallRecovery(
  messages: readonly CodexMessage[], dispatched: ReadonlySet<string>, partialAnswer: string,
): ChatGptStallRecovery | undefined {
  const calls = new Map<string, string | undefined>();
  const completed = new Map<string, CodexToolResultMessage>();
  const returned = new Set<string>();
  for (const message of messages) {
    if (message.role === "assistant") {
      for (const item of message.content) {
        if (item.type !== "toolCall" || !dispatched.has(item.id)) continue;
        calls.set(item.id, toolKey(namespacedToolName(item.namespace, item.name), item.arguments));
      }
    } else if (message.role === "toolResult" && dispatched.has(message.toolCallId)) {
      if (!calls.has(message.toolCallId) || returned.has(message.toolCallId)) return undefined;
      returned.add(message.toolCallId);
      const key = calls.get(message.toolCallId);
      if (key) completed.set(key, message);
    }
  }
  return returned.size === dispatched.size ? { completed, partialAnswer } : undefined;
}

export function findStallRecoveryResult(
  recovery: ChatGptStallRecovery, request: BrokerToolRequest,
): CodexToolResultMessage | undefined {
  const args = request.freeform ? { input: request.input ?? "" } : request.arguments ?? {};
  const key = toolKey(request.wireName, args);
  return key ? recovery.completed.get(key) : undefined;
}
