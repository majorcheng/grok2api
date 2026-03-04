import type { GrokSettings } from "../settings";
import { getDynamicHeaders } from "./headers";
import { getModelInfo, toGrokModel } from "./models";

export interface OpenAIToolDefinition {
  type?: string;
  function?: {
    name?: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface OpenAIToolCallMessage {
  id?: string;
  type?: string;
  function?: {
    name?: string;
    arguments?: unknown;
  };
}

export interface OpenAIChatMessage {
  role: string;
  content: string | Array<{ type: string; text?: string; image_url?: { url?: string } }> | null;
  tool_calls?: OpenAIToolCallMessage[];
  tool_call_id?: string;
}

export interface OpenAIChatRequestBody {
  model: string;
  messages: OpenAIChatMessage[];
  stream?: boolean;
  tools?: OpenAIToolDefinition[];
  tool_choice?: "auto" | "none" | "required" | { type?: string; function?: { name?: string } };
  parallel_tool_calls?: boolean;
  video_config?: {
    aspect_ratio?: string;
    video_length?: number;
    resolution?: string;
    preset?: string;
  };
}

export const CONVERSATION_API = "https://grok.com/rest/app-chat/conversations/new";

function toJsonText(value: unknown): string {
  if (typeof value === "string") {
    const text = value.trim();
    if (!text) return "{}";
    try {
      return JSON.stringify(JSON.parse(text));
    } catch {
      return text;
    }
  }
  if (value === undefined || value === null) return "{}";
  try {
    return JSON.stringify(value);
  } catch {
    return "{}";
  }
}

function collectToolNames(tools?: OpenAIToolDefinition[]): Set<string> {
  const names = new Set<string>();
  if (!Array.isArray(tools)) return names;
  for (const tool of tools) {
    const name = tool?.function?.name;
    if (typeof name === "string" && name.trim()) names.add(name.trim());
  }
  return names;
}

function buildToolInstruction(args: {
  tools?: OpenAIToolDefinition[];
  toolChoice?: OpenAIChatRequestBody["tool_choice"];
  parallelToolCalls?: boolean;
}): string {
  if (!Array.isArray(args.tools) || !args.tools.length) return "";
  if (args.toolChoice === "none") return "";

  const compactTools = args.tools
    .filter((tool) => tool?.type === "function")
    .map((tool) => {
      const fn = tool.function ?? {};
      return {
        type: "function",
        function: {
          name: String(fn.name ?? "").trim(),
          description: String(fn.description ?? ""),
          parameters:
            fn.parameters && typeof fn.parameters === "object"
              ? fn.parameters
              : { type: "object", properties: {} },
        },
      };
    })
    .filter((tool) => tool.function.name);
  if (!compactTools.length) return "";

  let choiceLine = "tool_choice=auto";
  if (typeof args.toolChoice === "string" && args.toolChoice.trim()) {
    choiceLine = `tool_choice=${args.toolChoice.trim()}`;
  } else if (args.toolChoice && typeof args.toolChoice === "object") {
    const forced = args.toolChoice.function?.name;
    if (typeof forced === "string" && forced.trim()) {
      choiceLine = `tool_choice=function:${forced.trim()}`;
    }
  }

  const parallelLine = args.parallelToolCalls === false ? "parallel_tool_calls=false" : "parallel_tool_calls=true";
  return [
    "[Tool Calling Contract]",
    "When tool use is needed, output ONLY valid JSON without markdown.",
    'JSON schema: {"tool_calls":[{"name":"<tool_name>","arguments":{...}}]}',
    "arguments must be a JSON object.",
    "If no tool is needed, output normal plain text.",
    `${choiceLine}; ${parallelLine}`,
    `tools=${JSON.stringify(compactTools)}`,
  ].join("\n");
}

export function extractContent(
  messages: OpenAIChatMessage[],
  opts?: {
    tools?: OpenAIToolDefinition[];
    tool_choice?: OpenAIChatRequestBody["tool_choice"];
    parallel_tool_calls?: boolean;
  },
): { content: string; images: string[] } {
  const images: string[] = [];
  const extracted: Array<{ role: string; text: string }> = [];
  const toolNames = collectToolNames(opts?.tools);

  for (const msg of messages) {
    const role = msg.role ?? "user";
    const content = msg.content ?? "";

    const parts: string[] = [];
    if (Array.isArray(content)) {
      for (const item of content) {
        if (item?.type === "text") {
          const t = item.text ?? "";
          if (t.trim()) parts.push(t);
        }
        if (item?.type === "image_url") {
          const url = item.image_url?.url;
          if (url) images.push(url);
        }
      }
    } else {
      const t = String(content);
      if (t.trim()) parts.push(t);
    }

    if (role === "assistant" && Array.isArray(msg.tool_calls)) {
      for (const call of msg.tool_calls) {
        const name = call?.function?.name;
        if (typeof name !== "string" || !name.trim()) continue;
        const cleanName = name.trim();
        if (toolNames.size && !toolNames.has(cleanName)) continue;
        const callId = typeof call.id === "string" && call.id.trim() ? call.id.trim() : "generated";
        const argsText = toJsonText(call?.function?.arguments);
        parts.push(`[assistant_tool_call id=${callId} name=${cleanName} arguments=${argsText}]`);
      }
    }

    if (role === "tool") {
      const toolCallId =
        typeof msg.tool_call_id === "string" && msg.tool_call_id.trim() ? msg.tool_call_id.trim() : "";
      const toolResult = parts.join("\n").trim();
      parts.length = 0;
      if (toolCallId) {
        if (toolResult) parts.push(`[tool_result id=${toolCallId}]\n${toolResult}`);
        else parts.push(`[tool_result id=${toolCallId}]`);
      }
    }

    if (parts.length) extracted.push({ role, text: parts.join("\n") });
  }

  let lastUserIndex: number | null = null;
  for (let i = extracted.length - 1; i >= 0; i--) {
    if (extracted[i]!.role === "user") {
      lastUserIndex = i;
      break;
    }
  }

  const out: string[] = [];
  for (let i = 0; i < extracted.length; i++) {
    const role = extracted[i]!.role || "user";
    const text = extracted[i]!.text;
    if (i === lastUserIndex) out.push(text);
    else out.push(`${role}: ${text}`);
  }

  const toolInstruction = buildToolInstruction({
    ...(opts?.tools ? { tools: opts.tools } : {}),
    ...(opts?.tool_choice !== undefined ? { toolChoice: opts.tool_choice } : {}),
    ...(opts?.parallel_tool_calls !== undefined ? { parallelToolCalls: opts.parallel_tool_calls } : {}),
  });
  if (toolInstruction) out.push(`system: ${toolInstruction}`);

  return { content: out.join("\n\n"), images };
}

export function buildConversationPayload(args: {
  requestModel: string;
  content: string;
  imgIds: string[];
  imgUris: string[];
  postId?: string;
  videoConfig?: {
    aspect_ratio?: string;
    video_length?: number;
    resolution?: string;
    preset?: string;
  };
  settings: GrokSettings;
}): { payload: Record<string, unknown>; referer?: string; isVideoModel: boolean } {
  const { requestModel, content, imgIds, imgUris, postId, settings } = args;
  const cfg = getModelInfo(requestModel);
  const { grokModel, mode, isVideoModel } = toGrokModel(requestModel);

  if (cfg?.is_video_model) {
    if (!postId) throw new Error("视频模型缺少 postId（需要先创建 media post）");

    const aspectRatio = (args.videoConfig?.aspect_ratio ?? "").trim() || "3:2";
    const videoLengthRaw = Number(args.videoConfig?.video_length ?? 6);
    const videoLength = Number.isFinite(videoLengthRaw) ? Math.max(1, Math.floor(videoLengthRaw)) : 6;
    const resolution = (args.videoConfig?.resolution ?? "SD") === "HD" ? "HD" : "SD";
    const preset = (args.videoConfig?.preset ?? "normal").trim();

    let modeFlag = "--mode=custom";
    if (preset === "fun") modeFlag = "--mode=extremely-crazy";
    else if (preset === "normal") modeFlag = "--mode=normal";
    else if (preset === "spicy") modeFlag = "--mode=extremely-spicy-or-crazy";

    const prompt = `${String(content || "").trim()} ${modeFlag}`.trim();

    return {
      isVideoModel: true,
      referer: "https://grok.com/imagine",
      payload: {
        temporary: true,
        modelName: "grok-3",
        message: prompt,
        toolOverrides: { videoGen: true },
        enableSideBySide: true,
        responseMetadata: {
          experiments: [],
          modelConfigOverride: {
            modelMap: {
              videoGenModelConfig: {
                parentPostId: postId,
                aspectRatio,
                videoLength,
                videoResolution: resolution,
              },
            },
          },
        },
      },
    };
  }

  return {
    isVideoModel,
    payload: {
      temporary: settings.temporary ?? true,
      modelName: grokModel,
      message: content,
      fileAttachments: imgIds,
      imageAttachments: [],
      disableSearch: false,
      enableImageGeneration: true,
      returnImageBytes: false,
      returnRawGrokInXaiRequest: false,
      enableImageStreaming: true,
      imageGenerationCount: 2,
      forceConcise: false,
      toolOverrides: {},
      enableSideBySide: true,
      sendFinalMetadata: true,
      isReasoning: false,
      webpageUrls: [],
      disableTextFollowUps: true,
      responseMetadata: { requestModelDetails: { modelId: grokModel } },
      disableMemory: false,
      forceSideBySide: false,
      modelMode: mode,
      isAsyncChat: false,
    },
  };
}

export async function sendConversationRequest(args: {
  payload: Record<string, unknown>;
  cookie: string;
  settings: GrokSettings;
  referer?: string;
}): Promise<Response> {
  const { payload, cookie, settings, referer } = args;
  const headers = getDynamicHeaders(settings, "/rest/app-chat/conversations/new");
  headers.Cookie = cookie;
  if (referer) headers.Referer = referer;
  const body = JSON.stringify(payload);

  return fetch(CONVERSATION_API, { method: "POST", headers, body });
}
