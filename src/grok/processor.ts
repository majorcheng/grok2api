import type { GrokSettings, GlobalSettings } from "../settings";

type GrokNdjson = Record<string, unknown>;
type FinishReason = "stop" | "error" | "tool_calls" | null;

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function readWithTimeout(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  ms: number,
): Promise<ReadableStreamReadResult<Uint8Array> | { timeout: true }> {
  if (ms <= 0) return { timeout: true };
  return Promise.race([
    reader.read(),
    sleep(ms).then(() => ({ timeout: true }) as const),
  ]);
}

function makeChunk(
  id: string,
  created: number,
  model: string,
  content: string,
  finish_reason?: FinishReason,
): string {
  return makeDeltaChunk(id, created, model, content ? { role: "assistant", content } : {}, finish_reason);
}

function makeDeltaChunk(
  id: string,
  created: number,
  model: string,
  delta: Record<string, unknown>,
  finish_reason?: FinishReason,
): string {
  const payload: Record<string, unknown> = {
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [
      {
        index: 0,
        delta,
        finish_reason: finish_reason ?? null,
      },
    ],
  };
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function makeDone(): string {
  return "data: [DONE]\n\n";
}

function toImgProxyUrl(globalCfg: GlobalSettings, origin: string, path: string): string {
  const baseUrl = (globalCfg.base_url ?? "").trim() || origin;
  return `${baseUrl}/images/${path}`;
}

function buildVideoTag(src: string): string {
  return `<video src="${src}" controls="controls" width="500" height="300"></video>\n`;
}

function buildVideoPosterPreview(videoUrl: string, posterUrl?: string): string {
  const href = String(videoUrl || "").replace(/"/g, "&quot;");
  const poster = String(posterUrl || "").replace(/"/g, "&quot;");
  if (!href) return "";
  if (!poster) return `<a href="${href}" target="_blank" rel="noopener noreferrer">${href}</a>\n`;
  return `<a href="${href}" target="_blank" rel="noopener noreferrer" style="display:inline-block;position:relative;max-width:100%;text-decoration:none;">
  <img src="${poster}" alt="video" style="max-width:100%;height:auto;border-radius:12px;display:block;" />
  <span style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">
    <span style="width:64px;height:64px;border-radius:9999px;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;">
      <span style="width:0;height:0;border-top:12px solid transparent;border-bottom:12px solid transparent;border-left:18px solid #fff;margin-left:4px;"></span>
    </span>
  </span>
</a>\n`;
}

function buildVideoHtml(args: { videoUrl: string; posterUrl?: string; posterPreview: boolean }): string {
  if (args.posterPreview) return buildVideoPosterPreview(args.videoUrl, args.posterUrl);
  return buildVideoTag(args.videoUrl);
}

function base64UrlEncode(input: string): string {
  const bytes = new TextEncoder().encode(input);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function encodeAssetPath(raw: string): string {
  try {
    const u = new URL(raw);
    // Keep full URL (query etc.) to avoid lossy pathname-only encoding (some URLs may encode the real path in query).
    return `u_${base64UrlEncode(u.toString())}`;
  } catch {
    const p = raw.startsWith("/") ? raw : `/${raw}`;
    return `p_${base64UrlEncode(p)}`;
  }
}

function normalizeGeneratedAssetUrls(input: unknown): string[] {
  if (!Array.isArray(input)) return [];

  const out: string[] = [];
  for (const v of input) {
    if (typeof v !== "string") continue;
    const s = v.trim();
    if (!s) continue;
    if (s === "/") continue;

    try {
      const u = new URL(s);
      if (u.pathname === "/" && !u.search && !u.hash) continue;
    } catch {
      // ignore (path-style strings are allowed)
    }

    out.push(s);
  }

  return out;
}

function normalizeToolNames(tools?: unknown[]): Set<string> {
  const names = new Set<string>();
  if (!Array.isArray(tools)) return names;
  for (const tool of tools) {
    if (!tool || typeof tool !== "object") continue;
    const fn = (tool as Record<string, unknown>).function;
    if (!fn || typeof fn !== "object") continue;
    const name = (fn as Record<string, unknown>).name;
    if (typeof name === "string" && name.trim()) names.add(name.trim());
  }
  return names;
}

function toArgumentsString(value: unknown): string {
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

function extractJsonCandidates(text: string): string[] {
  const raw = String(text || "").trim();
  if (!raw) return [];

  const out: string[] = [raw];
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)\s*```/gi) ?? [];
  for (const block of fenced) {
    const stripped = block
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/i, "")
      .trim();
    if (stripped) out.push(stripped);
  }

  const firstObj = raw.indexOf("{");
  const lastObj = raw.lastIndexOf("}");
  if (firstObj >= 0 && lastObj > firstObj) out.push(raw.slice(firstObj, lastObj + 1).trim());

  const firstArr = raw.indexOf("[");
  const lastArr = raw.lastIndexOf("]");
  if (firstArr >= 0 && lastArr > firstArr) out.push(raw.slice(firstArr, lastArr + 1).trim());

  const deduped: string[] = [];
  const seen = new Set<string>();
  for (const item of out) {
    if (!item || seen.has(item)) continue;
    seen.add(item);
    deduped.push(item);
  }
  return deduped;
}

function normalizeToolCall(item: unknown, allowedNames: Set<string>): Record<string, unknown> | null {
  if (!item || typeof item !== "object") return null;
  const rec = item as Record<string, unknown>;

  const fnRaw = rec.function;
  const fn = fnRaw && typeof fnRaw === "object" ? (fnRaw as Record<string, unknown>) : null;

  const nameRaw = fn ? fn.name : rec.name;
  if (typeof nameRaw !== "string" || !nameRaw.trim()) return null;
  const name = nameRaw.trim();
  if (allowedNames.size && !allowedNames.has(name)) return null;

  const argsRaw = fn ? fn.arguments : rec.arguments;
  const callIdRaw = rec.id;
  const callId = typeof callIdRaw === "string" && callIdRaw.trim() ? callIdRaw.trim() : `call_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;

  return {
    id: callId,
    type: "function",
    function: {
      name,
      arguments: toArgumentsString(argsRaw),
    },
  };
}

function extractToolCallsFromText(text: string, tools?: unknown[]): Record<string, unknown>[] | null {
  const allowedNames = normalizeToolNames(tools);
  if (!allowedNames.size) return null;

  for (const candidate of extractJsonCandidates(text)) {
    let payload: unknown;
    try {
      payload = JSON.parse(candidate);
    } catch {
      continue;
    }

    let calls: unknown[] = [];
    if (Array.isArray(payload)) {
      calls = payload;
    } else if (payload && typeof payload === "object") {
      const rec = payload as Record<string, unknown>;
      if (Array.isArray(rec.tool_calls)) calls = rec.tool_calls;
      else if ("name" in rec || "function" in rec) calls = [rec];
    }
    if (!calls.length) continue;

    const normalized: Record<string, unknown>[] = [];
    for (const item of calls) {
      const call = normalizeToolCall(item, allowedNames);
      if (call) normalized.push(call);
    }
    if (normalized.length) return normalized;
  }
  return null;
}

export function createOpenAiStreamFromGrokNdjson(
  grokResp: Response,
  opts: {
    cookie: string;
    settings: GrokSettings;
    global: GlobalSettings;
    origin: string;
    requestedModel: string;
    tools?: unknown[];
    onFinish?: (result: { status: number; duration: number }) => Promise<void> | void;
  },
): ReadableStream<Uint8Array> {
  const { settings, global, origin } = opts;
  const fallbackModel =
    typeof opts.requestedModel === "string" && opts.requestedModel.trim()
      ? opts.requestedModel.trim()
      : "grok-4";
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();

  const id = `chatcmpl-${crypto.randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);

  const filteredTags = (settings.filtered_tags ?? "")
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  const showThinking = settings.show_thinking !== false;
  const enableToolCalls = Array.isArray(opts.tools) && opts.tools.length > 0;
  const bufferFinalText = enableToolCalls;

  const firstTimeoutMs = Math.max(0, (settings.stream_first_response_timeout ?? 30) * 1000);
  const chunkTimeoutMs = Math.max(0, (settings.stream_chunk_timeout ?? 120) * 1000);
  const totalTimeoutMs = Math.max(0, (settings.stream_total_timeout ?? 600) * 1000);

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const body = grokResp.body;
      if (!body) {
        controller.enqueue(encoder.encode(makeChunk(id, created, fallbackModel, "Empty response", "error")));
        controller.enqueue(encoder.encode(makeDone()));
        controller.close();
        return;
      }

      const reader = body.getReader();
      const startTime = Date.now();
      let finalStatus = 200;
      let lastChunkTime = startTime;
      let firstReceived = false;

      let currentModel = fallbackModel;
      let isImage = false;
      let isThinking = false;
      let thinkingFinished = false;
      let videoProgressStarted = false;
      let lastVideoProgress = -1;
      const bufferedTokens: string[] = [];

      let buffer = "";

      const flushStop = () => {
        controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, "", "stop")));
        controller.enqueue(encoder.encode(makeDone()));
      };

      try {
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const now = Date.now();
          const elapsed = now - startTime;
          if (!firstReceived && elapsed > firstTimeoutMs) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }
          if (totalTimeoutMs > 0 && elapsed > totalTimeoutMs) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }
          const idle = now - lastChunkTime;
          if (firstReceived && idle > chunkTimeoutMs) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }

          const perReadTimeout = Math.min(
            firstReceived ? chunkTimeoutMs : firstTimeoutMs,
            totalTimeoutMs > 0 ? Math.max(0, totalTimeoutMs - elapsed) : Number.POSITIVE_INFINITY,
          );

          const res = await readWithTimeout(reader, perReadTimeout);
          if ("timeout" in res) {
            flushStop();
            if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
            controller.close();
            return;
          }

          const { value, done } = res;
          if (done) break;
          if (!value) continue;
          buffer += decoder.decode(value, { stream: true });

          let idx: number;
          while ((idx = buffer.indexOf("\n")) !== -1) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (!line) continue;

            let data: GrokNdjson;
            try {
              data = JSON.parse(line) as GrokNdjson;
            } catch {
              continue;
            }

            firstReceived = true;
            lastChunkTime = Date.now();

            const err = (data as any).error;
            if (err?.message) {
              finalStatus = 500;
              controller.enqueue(
                encoder.encode(makeChunk(id, created, currentModel, `Error: ${String(err.message)}`, "stop")),
              );
              controller.enqueue(encoder.encode(makeDone()));
              if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
              controller.close();
              return;
            }

            const grok = (data as any).result?.response;
            if (!grok) continue;

            const userRespModel = grok.userResponse?.model;
            if (typeof userRespModel === "string" && userRespModel.trim()) currentModel = userRespModel.trim();

            // Video generation stream
            const videoResp = grok.streamingVideoGenerationResponse;
            if (videoResp) {
              const progress = typeof videoResp.progress === "number" ? videoResp.progress : 0;
              const videoUrl = typeof videoResp.videoUrl === "string" ? videoResp.videoUrl : "";
              const thumbUrl = typeof videoResp.thumbnailImageUrl === "string" ? videoResp.thumbnailImageUrl : "";

              if (progress > lastVideoProgress) {
                lastVideoProgress = progress;
                if (showThinking) {
                  let msg = "";
                  if (!videoProgressStarted) {
                    msg = `<think>视频已生成${progress}%\n`;
                    videoProgressStarted = true;
                  } else if (progress < 100) {
                    msg = `视频已生成${progress}%\n`;
                  } else {
                    msg = `视频已生成${progress}%</think>\n`;
                  }
                  controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, msg)));
                }
              }

              if (videoUrl) {
                const videoPath = encodeAssetPath(videoUrl);
                const src = toImgProxyUrl(global, origin, videoPath);

                let poster: string | undefined;
                if (thumbUrl) {
                  const thumbPath = encodeAssetPath(thumbUrl);
                  poster = toImgProxyUrl(global, origin, thumbPath);
                }

                controller.enqueue(
                  encoder.encode(
                    makeChunk(
                      id,
                      created,
                      currentModel,
                      buildVideoHtml({
                        videoUrl: src,
                        posterPreview: settings.video_poster_preview === true,
                        ...(poster ? { posterUrl: poster } : {}),
                      }),
                    ),
                  ),
                );
              }
              continue;
            }

            if (grok.imageAttachmentInfo) isImage = true;
            const rawToken = grok.token;

            if (isImage) {
              const modelResp = grok.modelResponse;
              if (modelResp) {
                const urls = normalizeGeneratedAssetUrls(modelResp.generatedImageUrls);
                if (urls.length) {
                  const linesOut: string[] = [];
                  for (const u of urls) {
                    const imgPath = encodeAssetPath(u);
                    const imgUrl = toImgProxyUrl(global, origin, imgPath);
                    linesOut.push(`![Generated Image](${imgUrl})`);
                  }
                  controller.enqueue(
                    encoder.encode(makeChunk(id, created, currentModel, linesOut.join("\n"), "stop")),
                  );
                  controller.enqueue(encoder.encode(makeDone()));
                  if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
                  controller.close();
                  return;
                }
              } else if (typeof rawToken === "string" && rawToken) {
                controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, rawToken)));
              }
              continue;
            }

            // Text chat stream
            if (Array.isArray(rawToken)) continue;
            if (typeof rawToken !== "string" || !rawToken) continue;
            let token = rawToken;

            if (filteredTags.some((t) => token.includes(t))) continue;

            const currentIsThinking = Boolean(grok.isThinking);
            const messageTag = grok.messageTag;

            if (thinkingFinished && currentIsThinking) continue;

            if (grok.toolUsageCardId && grok.webSearchResults?.results && Array.isArray(grok.webSearchResults.results)) {
              if (currentIsThinking) {
                if (showThinking) {
                  let appended = "";
                  for (const r of grok.webSearchResults.results) {
                    const title = typeof r.title === "string" ? r.title : "";
                    const url = typeof r.url === "string" ? r.url : "";
                    const preview = typeof r.preview === "string" ? r.preview.replace(/\n/g, "") : "";
                    appended += `\n- [${title}](${url} \"${preview}\")`;
                  }
                  token += `${appended}\n`;
                } else {
                  continue;
                }
              } else {
                continue;
              }
            }

            let content = token;
            if (messageTag === "header") content = `\n\n${token}\n\n`;

            let shouldSkip = false;
            if (!isThinking && currentIsThinking) {
              if (showThinking) content = `<think>\n${content}`;
              else shouldSkip = true;
            } else if (isThinking && !currentIsThinking) {
              if (showThinking) content = `\n</think>\n${content}`;
              thinkingFinished = true;
            } else if (currentIsThinking && !showThinking) {
              shouldSkip = true;
            }

            if (!shouldSkip) {
              if (bufferFinalText) bufferedTokens.push(content);
              else controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, content)));
            }
            isThinking = currentIsThinking;
          }
        }

        if (bufferFinalText) {
          const fullText = bufferedTokens.join("").trim();
          const toolCalls = enableToolCalls ? extractToolCallsFromText(fullText, opts.tools) : null;
          if (toolCalls?.length) {
            controller.enqueue(encoder.encode(makeDeltaChunk(id, created, currentModel, { role: "assistant" }, null)));
            toolCalls.forEach((call, index) => {
              const fn = (call.function ?? {}) as Record<string, unknown>;
              controller.enqueue(
                encoder.encode(
                  makeDeltaChunk(
                    id,
                    created,
                    currentModel,
                    {
                      tool_calls: [
                        {
                          index,
                          id: call.id,
                          type: "function",
                          function: { name: fn.name, arguments: "" },
                        },
                      ],
                    },
                    null,
                  ),
                ),
              );
              controller.enqueue(
                encoder.encode(
                  makeDeltaChunk(
                    id,
                    created,
                    currentModel,
                    {
                      tool_calls: [
                        {
                          index,
                          function: { arguments: fn.arguments },
                        },
                      ],
                    },
                    null,
                  ),
                ),
              );
            });
            controller.enqueue(encoder.encode(makeDeltaChunk(id, created, currentModel, {}, "tool_calls")));
          } else {
            const finalText = fullText;
            if (finalText) controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, finalText)));
            controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, "", "stop")));
          }
        } else {
          controller.enqueue(encoder.encode(makeChunk(id, created, currentModel, "", "stop")));
        }
        controller.enqueue(encoder.encode(makeDone()));
        if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
        controller.close();
      } catch (e) {
        finalStatus = 500;
        controller.enqueue(
          encoder.encode(
            makeChunk(id, created, currentModel, `处理错误: ${e instanceof Error ? e.message : String(e)}`, "error"),
          ),
        );
        controller.enqueue(encoder.encode(makeDone()));
        if (opts.onFinish) await opts.onFinish({ status: finalStatus, duration: (Date.now() - startTime) / 1000 });
        controller.close();
      } finally {
        try {
          reader.releaseLock();
        } catch {
          // ignore
        }
      }
    },
  });
}

export async function parseOpenAiFromGrokNdjson(
  grokResp: Response,
  opts: {
    cookie: string;
    settings: GrokSettings;
    global: GlobalSettings;
    origin: string;
    requestedModel: string;
    tools?: unknown[];
  },
): Promise<Record<string, unknown>> {
  const { global, origin, requestedModel, settings } = opts;
  const text = await grokResp.text();
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);

  let content = "";
  let model = requestedModel;
  for (const line of lines) {
    let data: GrokNdjson;
    try {
      data = JSON.parse(line) as GrokNdjson;
    } catch {
      continue;
    }

    const err = (data as any).error;
    if (err?.message) throw new Error(String(err.message));

    const grok = (data as any).result?.response;
    if (!grok) continue;

    const videoResp = grok.streamingVideoGenerationResponse;
    if (videoResp?.videoUrl && typeof videoResp.videoUrl === "string") {
      const videoPath = encodeAssetPath(videoResp.videoUrl);
      const src = toImgProxyUrl(global, origin, videoPath);

      let poster: string | undefined;
      if (typeof videoResp.thumbnailImageUrl === "string" && videoResp.thumbnailImageUrl) {
        const thumbPath = encodeAssetPath(videoResp.thumbnailImageUrl);
        poster = toImgProxyUrl(global, origin, thumbPath);
      }

      content = buildVideoHtml({
        videoUrl: src,
        posterPreview: settings.video_poster_preview === true,
        ...(poster ? { posterUrl: poster } : {}),
      });
      model = requestedModel;
      break;
    }

    const modelResp = grok.modelResponse;
    if (!modelResp) continue;
    if (typeof modelResp.error === "string" && modelResp.error) throw new Error(modelResp.error);

    if (typeof modelResp.model === "string" && modelResp.model) model = modelResp.model;
    if (typeof modelResp.message === "string") content = modelResp.message;

    const rawUrls = modelResp.generatedImageUrls;
    const urls = normalizeGeneratedAssetUrls(rawUrls);
    if (urls.length) {
      for (const u of urls) {
        const imgPath = encodeAssetPath(u);
        const imgUrl = toImgProxyUrl(global, origin, imgPath);
        content += `\n![Generated Image](${imgUrl})`;
      }
      break;
    }

    // If upstream emits placeholder/empty generatedImageUrls in intermediate frames, keep scanning.
    if (Array.isArray(rawUrls)) continue;

    // For normal chat replies, the first modelResponse is enough.
    break;
  }

  const toolCalls = extractToolCallsFromText(content, opts.tools);
  const message = toolCalls
    ? { role: "assistant", content: null, tool_calls: toolCalls }
    : { role: "assistant", content };
  const finishReason: FinishReason = toolCalls ? "tool_calls" : "stop";

  return {
    id: `chatcmpl-${crypto.randomUUID()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: finishReason,
      },
    ],
    usage: null,
  };
}
