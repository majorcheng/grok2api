"""
OpenAI 响应格式处理器
"""
import time
import uuid
import random
import html
import json
import re
import orjson
from typing import Any, AsyncGenerator, Optional, AsyncIterable, List

from app.core.config import get_config
from app.core.logger import logger
from app.services.grok.assets import DownloadService


ASSET_URL = "https://assets.grok.com/"


def _build_video_poster_preview(video_url: str, thumbnail_url: str = "") -> str:
    """将 <video> 替换为可点击的 Poster 预览图（用于前端展示）"""
    safe_video = html.escape(video_url or "", quote=True)
    safe_thumb = html.escape(thumbnail_url or "", quote=True)

    if not safe_video:
        return ""

    if not safe_thumb:
        return f'<a href="{safe_video}" target="_blank" rel="noopener noreferrer">{safe_video}</a>'

    return f'''<a href="{safe_video}" target="_blank" rel="noopener noreferrer" style="display:inline-block;position:relative;max-width:100%;text-decoration:none;">
  <img src="{safe_thumb}" alt="video" style="max-width:100%;height:auto;border-radius:12px;display:block;" />
  <span style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">
    <span style="width:64px;height:64px;border-radius:9999px;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;">
      <span style="width:0;height:0;border-top:12px solid transparent;border-bottom:12px solid transparent;border-left:18px solid #fff;margin-left:4px;"></span>
    </span>
  </span>
</a>'''


def _safe_json_loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return None


def _iter_json_candidates(text: str) -> List[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []

    candidates: List[str] = [stripped]

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    for block in fenced:
        block = block.strip()
        if block:
            candidates.append(block)

    first_obj = stripped.find("{")
    last_obj = stripped.rfind("}")
    if first_obj != -1 and last_obj > first_obj:
        part = stripped[first_obj:last_obj + 1].strip()
        if part:
            candidates.append(part)

    first_arr = stripped.find("[")
    last_arr = stripped.rfind("]")
    if first_arr != -1 and last_arr > first_arr:
        part = stripped[first_arr:last_arr + 1].strip()
        if part:
            candidates.append(part)

    # 去重并保持顺序
    seen = set()
    deduped: List[str] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _to_arguments_string(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "{}"
        parsed = _safe_json_loads(text)
        if parsed is not None:
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        return text
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _extract_tool_names(tools: Optional[List[dict[str, Any]]]) -> set[str]:
    names = set()
    if not tools:
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def _normalize_tool_call(item: Any, allowed_names: set[str]) -> Optional[dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    fn = item.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        arguments = fn.get("arguments")
        call_id = item.get("id")
    else:
        name = item.get("name")
        arguments = item.get("arguments")
        call_id = item.get("id")

    if not isinstance(name, str) or not name.strip():
        return None

    clean_name = name.strip()
    if allowed_names and clean_name not in allowed_names:
        return None

    if not isinstance(call_id, str) or not call_id.strip():
        call_id = f"call_{uuid.uuid4().hex[:24]}"

    return {
        "id": call_id.strip(),
        "type": "function",
        "function": {
            "name": clean_name,
            "arguments": _to_arguments_string(arguments)
        }
    }


def extract_tool_calls_from_text(
    text: str,
    tools: Optional[List[dict[str, Any]]] = None
) -> Optional[List[dict[str, Any]]]:
    """从模型文本中识别 OpenAI tool_calls 结构"""
    allowed_names = _extract_tool_names(tools)
    if not allowed_names:
        return None

    for candidate in _iter_json_candidates(text):
        payload = _safe_json_loads(candidate)
        if payload is None:
            continue

        calls: list[Any] = []
        if isinstance(payload, dict):
            raw_calls = payload.get("tool_calls")
            if isinstance(raw_calls, list):
                calls = raw_calls
            elif "name" in payload or "function" in payload:
                calls = [payload]
        elif isinstance(payload, list):
            calls = payload

        if not calls:
            continue

        normalized: List[dict[str, Any]] = []
        for item in calls:
            call = _normalize_tool_call(item, allowed_names)
            if call:
                normalized.append(call)

        if normalized:
            return normalized

    return None


def normalize_response_format_type(response_format: Any) -> str:
    if response_format is None:
        return "text"
    if isinstance(response_format, str):
        rf = response_format.strip().lower()
        return rf if rf in {"text", "json_object", "json_schema"} else "text"
    if isinstance(response_format, dict):
        rf = response_format.get("type")
        if isinstance(rf, str):
            rf = rf.strip().lower()
            if rf in {"text", "json_object", "json_schema"}:
                return rf
    return "text"


def extract_json_value_from_text(text: str, expect_object: bool = False) -> Any:
    for candidate in _iter_json_candidates(text):
        payload = _safe_json_loads(candidate)
        if payload is None:
            continue
        if expect_object and not isinstance(payload, dict):
            continue
        return payload
    return None


def enforce_json_response_text(text: str, response_format: Any) -> str:
    """将输出收敛为 JSON 文本，避免返回额外拒绝话术/markdown。"""
    rf = normalize_response_format_type(response_format)
    if rf not in {"json_object", "json_schema"}:
        return text

    raw = (text or "").strip()
    if rf == "json_object":
        payload = extract_json_value_from_text(raw, expect_object=True)
        if payload is None:
            payload = {"output": raw}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    # json_schema: best-effort，不做 schema 校验（避免额外依赖）
    payload = extract_json_value_from_text(raw, expect_object=False)
    if payload is None:
        payload = {"output": raw}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class BaseProcessor:
    """基础处理器"""
    
    def __init__(self, model: str, token: str = ""):
        self.model = model
        self.token = token
        self.created = int(time.time())
        self.app_url = get_config("app.app_url", "")
        self._dl_service: Optional[DownloadService] = None

    def _get_dl(self) -> DownloadService:
        """获取下载服务实例（复用）"""
        if self._dl_service is None:
            self._dl_service = DownloadService()
        return self._dl_service

    async def close(self):
        """释放下载服务资源"""
        if self._dl_service:
            await self._dl_service.close()
            self._dl_service = None

    async def process_url(self, path: str, media_type: str = "image") -> str:
        """处理资产 URL"""
        # 处理可能的绝对路径
        if path.startswith("http"):
            from urllib.parse import urlparse
            path = urlparse(path).path
            
        if not path.startswith("/"):
            path = f"/{path}"

        # Invalid root path is not a displayable image URL.
        if path in {"", "/"}:
            return ""

        # Always materialize to local cache endpoint so callers don't rely on
        # direct assets.grok.com access (often blocked without upstream cookies).
        dl_service = self._get_dl()
        await dl_service.download(path, self.token, media_type)
        local_path = f"/v1/files/{media_type}{path}"
        if self.app_url:
            return f"{self.app_url.rstrip('/')}{local_path}"
        return local_path

    def _sse_delta(self, delta: dict[str, Any], finish: str = None) -> str:
        """构建自定义 delta 的 SSE 响应"""
        if not hasattr(self, 'response_id'):
            self.response_id = None
        if not hasattr(self, 'fingerprint'):
            self.fingerprint = ""

        chunk = {
            "id": self.response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.fingerprint if hasattr(self, 'fingerprint') else "",
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]
        }
        return f"data: {orjson.dumps(chunk).decode()}\n\n"
            
    def _sse(self, content: str = "", role: str = None, finish: str = None) -> str:
        """构建 SSE 响应 (StreamProcessor 通用)"""
        delta = {}
        if role:
            delta["role"] = role
            delta["content"] = ""
        elif content:
            delta["content"] = content
        return self._sse_delta(delta, finish)


class StreamProcessor(BaseProcessor):
    """流式响应处理器"""
    
    def __init__(
        self,
        model: str,
        token: str = "",
        think: bool = None,
        response_format: Any = None,
        tools: Optional[List[dict[str, Any]]] = None
    ):
        super().__init__(model, token)
        self.response_id: Optional[str] = None
        self.fingerprint: str = ""
        self.think_opened: bool = False
        self.role_sent: bool = False
        self.filter_tags = get_config("grok.filter_tags", [])
        self.image_format = get_config("app.image_format", "url")
        self.response_format = response_format
        self.response_format_type = normalize_response_format_type(response_format)
        self.tools = tools or []
        self.enable_tool_calls = bool(self.tools)
        self.enforce_json_output = self.response_format_type in {"json_object", "json_schema"}
        self.buffer_final_text = self.enable_tool_calls or self.enforce_json_output
        self._buffered_tokens: List[str] = []
        self._has_media_output: bool = False
        
        if think is None:
            self.show_think = get_config("grok.thinking", False)
        else:
            self.show_think = think
    
    async def process(self, response: AsyncIterable[bytes]) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                # 元数据
                if (llm := resp.get("llmInfo")) and not self.fingerprint:
                    self.fingerprint = llm.get("modelHash", "")
                if rid := resp.get("responseId"):
                    self.response_id = rid
                
                # 首次发送 role
                if not self.role_sent:
                    yield self._sse(role="assistant")
                    self.role_sent = True
                
                # 图像生成进度
                if img := resp.get("streamingImageGenerationResponse"):
                    if self.show_think:
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                        idx = img.get('imageIndex', 0) + 1
                        progress = img.get('progress', 0)
                        yield self._sse(f"正在生成第{idx}张图片中，当前进度{progress}%\n")
                    continue
                
                # modelResponse
                if mr := resp.get("modelResponse"):
                    if self.think_opened and self.show_think:
                        if msg := mr.get("message"):
                            if self.buffer_final_text:
                                self._buffered_tokens.append(str(msg) + "\n")
                            else:
                                yield self._sse(msg + "\n")
                        if self.buffer_final_text:
                            self._buffered_tokens.append("</think>\n")
                        else:
                            yield self._sse("</think>\n")
                        self.think_opened = False
                    
                    # 处理生成的图片
                    for url in mr.get("generatedImageUrls", []):
                        self._has_media_output = True
                        parts = url.split("/")
                        img_id = parts[-2] if len(parts) >= 2 else "image"
                        
                        if self.image_format == "base64":
                            dl_service = self._get_dl()
                            base64_data = await dl_service.to_base64(url, self.token, "image")
                            if base64_data:
                                yield self._sse(f"![{img_id}]({base64_data})\n")
                            else:
                                final_url = await self.process_url(url, "image")
                                yield self._sse(f"![{img_id}]({final_url})\n")
                        else:
                            final_url = await self.process_url(url, "image")
                            yield self._sse(f"![{img_id}]({final_url})\n")
                    
                    if (meta := mr.get("metadata", {})).get("llm_info", {}).get("modelHash"):
                        self.fingerprint = meta["llm_info"]["modelHash"]
                    continue
                
                # 普通 token
                if (token := resp.get("token")) is not None:
                    if token and not (self.filter_tags and any(t in token for t in self.filter_tags)):
                        if self.buffer_final_text:
                            self._buffered_tokens.append(str(token))
                        else:
                            yield self._sse(token)
                        
            if self.think_opened:
                if self.buffer_final_text:
                    self._buffered_tokens.append("</think>\n")
                else:
                    yield self._sse("</think>\n")

            if self.buffer_final_text:
                tool_calls = extract_tool_calls_from_text("".join(self._buffered_tokens).strip(), self.tools)
                if tool_calls:
                    for idx, call in enumerate(tool_calls):
                        yield self._sse_delta({
                            "tool_calls": [{
                                "index": idx,
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["function"]["name"],
                                    "arguments": ""
                                }
                            }]
                        })
                        yield self._sse_delta({
                            "tool_calls": [{
                                "index": idx,
                                "function": {
                                    "arguments": call["function"]["arguments"]
                                }
                            }]
                        })
                    yield self._sse_delta({}, finish="tool_calls")
                else:
                    final_text = "".join(self._buffered_tokens).strip()
                    if self.enforce_json_output and not self._has_media_output:
                        final_text = enforce_json_response_text(final_text, self.response_format)
                    if final_text:
                        yield self._sse(final_text)
                    yield self._sse(finish="stop")
            else:
                yield self._sse(finish="stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream processing error: {e}", extra={"model": self.model})
            raise
        finally:
            await self.close()


class CollectProcessor(BaseProcessor):
    """非流式响应处理器"""
    
    def __init__(
        self,
        model: str,
        token: str = "",
        response_format: Any = None,
        tools: Optional[List[dict[str, Any]]] = None
    ):
        super().__init__(model, token)
        self.image_format = get_config("app.image_format", "url")
        self.response_format = response_format
        self.response_format_type = normalize_response_format_type(response_format)
        self.enforce_json_output = self.response_format_type in {"json_object", "json_schema"}
        self.tools = tools or []
    
    async def process(self, response: AsyncIterable[bytes]) -> dict[str, Any]:
        """处理并收集完整响应"""
        response_id = ""
        fingerprint = ""
        content = ""
        has_media_output = False
        
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                if (llm := resp.get("llmInfo")) and not fingerprint:
                    fingerprint = llm.get("modelHash", "")
                
                if mr := resp.get("modelResponse"):
                    response_id = mr.get("responseId", "")
                    content = mr.get("message", "")
                    
                    if urls := mr.get("generatedImageUrls"):
                        has_media_output = True
                        content += "\n"
                        for url in urls:
                            parts = url.split("/")
                            img_id = parts[-2] if len(parts) >= 2 else "image"
                            
                            if self.image_format == "base64":
                                dl_service = self._get_dl()
                                base64_data = await dl_service.to_base64(url, self.token, "image")
                                if base64_data:
                                    content += f"![{img_id}]({base64_data})\n"
                                else:
                                    final_url = await self.process_url(url, "image")
                                    content += f"![{img_id}]({final_url})\n"
                            else:
                                final_url = await self.process_url(url, "image")
                                content += f"![{img_id}]({final_url})\n"
                    
                    if (meta := mr.get("metadata", {})).get("llm_info", {}).get("modelHash"):
                        fingerprint = meta["llm_info"]["modelHash"]
                            
        except Exception as e:
            logger.error(f"Collect processing error: {e}", extra={"model": self.model})
        finally:
            await self.close()

        message_content: Optional[str] = content
        finish_reason = "stop"
        message_payload: dict[str, Any] = {
            "role": "assistant",
            "content": message_content,
            "refusal": None,
            "annotations": []
        }

        tool_calls = extract_tool_calls_from_text(content, self.tools)
        if tool_calls:
            finish_reason = "tool_calls"
            message_payload["content"] = None
            message_payload["tool_calls"] = tool_calls
        elif self.enforce_json_output and not has_media_output:
            message_payload["content"] = enforce_json_response_text(content, self.response_format)

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": fingerprint,
            "choices": [{
                "index": 0,
                "message": message_payload,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0, "image_tokens": 0},
                "completion_tokens_details": {"text_tokens": 0, "audio_tokens": 0, "reasoning_tokens": 0}
            }
        }


class VideoStreamProcessor(BaseProcessor):
    """视频流式响应处理器"""
    
    def __init__(self, model: str, token: str = "", think: bool = None):
        super().__init__(model, token)
        self.response_id: Optional[str] = None
        self.think_opened: bool = False
        self.role_sent: bool = False
        self.video_format = get_config("app.video_format", "url")
        
        if think is None:
            self.show_think = get_config("grok.thinking", False)
        else:
            self.show_think = think
    
    def _build_video_html(self, video_url: str, thumbnail_url: str = "") -> str:
        """构建视频 HTML 标签"""
        if get_config("grok.video_poster_preview", False):
            return _build_video_poster_preview(video_url, thumbnail_url)
        poster_attr = f' poster="{thumbnail_url}"' if thumbnail_url else ""
        return f'''<video id="video" controls="" preload="none"{poster_attr}>
  <source id="mp4" src="{video_url}" type="video/mp4">
</video>'''
    
    async def process(self, response: AsyncIterable[bytes]) -> AsyncGenerator[str, None]:
        """处理视频流式响应"""
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                if rid := resp.get("responseId"):
                    self.response_id = rid
                
                # 首次发送 role
                if not self.role_sent:
                    yield self._sse(role="assistant")
                    self.role_sent = True
                
                # 视频生成进度
                if video_resp := resp.get("streamingVideoGenerationResponse"):
                    progress = video_resp.get("progress", 0)
                    
                    if self.show_think:
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                        yield self._sse(f"正在生成视频中，当前进度{progress}%\n")
                    
                    if progress == 100:
                        video_url = video_resp.get("videoUrl", "")
                        thumbnail_url = video_resp.get("thumbnailImageUrl", "")
                        
                        if self.think_opened and self.show_think:
                            yield self._sse("</think>\n")
                            self.think_opened = False
                        
                        if video_url:
                            final_video_url = await self.process_url(video_url, "video")
                            final_thumbnail_url = ""
                            if thumbnail_url:
                                final_thumbnail_url = await self.process_url(thumbnail_url, "image")
                            
                            video_html = self._build_video_html(final_video_url, final_thumbnail_url)
                            yield self._sse(video_html)
                            
                            logger.info(f"Video generated: {video_url}")
                    continue
                        
            if self.think_opened:
                yield self._sse("</think>\n")
            yield self._sse(finish="stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Video stream processing error: {e}", extra={"model": self.model})
        finally:
            await self.close()


class VideoCollectProcessor(BaseProcessor):
    """视频非流式响应处理器"""
    
    def __init__(self, model: str, token: str = ""):
        super().__init__(model, token)
        self.video_format = get_config("app.video_format", "url")
    
    def _build_video_html(self, video_url: str, thumbnail_url: str = "") -> str:
        if get_config("grok.video_poster_preview", False):
            return _build_video_poster_preview(video_url, thumbnail_url)
        poster_attr = f' poster="{thumbnail_url}"' if thumbnail_url else ""
        return f'''<video id="video" controls="" preload="none"{poster_attr}>
  <source id="mp4" src="{video_url}" type="video/mp4">
</video>'''
    
    async def process(self, response: AsyncIterable[bytes]) -> dict[str, Any]:
        """处理并收集视频响应"""
        response_id = ""
        content = ""
        
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                if video_resp := resp.get("streamingVideoGenerationResponse"):
                    if video_resp.get("progress") == 100:
                        response_id = resp.get("responseId", "")
                        video_url = video_resp.get("videoUrl", "")
                        thumbnail_url = video_resp.get("thumbnailImageUrl", "")
                        
                        if video_url:
                            final_video_url = await self.process_url(video_url, "video")
                            final_thumbnail_url = ""
                            if thumbnail_url:
                                final_thumbnail_url = await self.process_url(thumbnail_url, "image")
                            
                            content = self._build_video_html(final_video_url, final_thumbnail_url)
                            logger.info(f"Video generated: {video_url}")
                            
        except Exception as e:
            logger.error(f"Video collect processing error: {e}", extra={"model": self.model})
        finally:
            await self.close()
        
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": None},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }


class ImageStreamProcessor(BaseProcessor):
    """图片生成流式响应处理器"""
    
    def __init__(
        self,
        model: str,
        token: str = "",
        n: int = 1,
        response_format: str = "b64_json",
    ):
        super().__init__(model, token)
        self.partial_index = 0
        self.n = n
        self.target_index = random.randint(0, 1) if n == 1 else None
        self.response_format = (response_format or "b64_json").lower()
        if self.response_format == "url":
            self.response_field = "url"
        elif self.response_format == "base64":
            self.response_field = "base64"
        else:
            self.response_field = "b64_json"
    
    def _sse(self, event: str, data: dict) -> str:
        """构建 SSE 响应 (覆盖基类)"""
        return f"event: {event}\ndata: {orjson.dumps(data).decode()}\n\n"
    
    async def process(self, response: AsyncIterable[bytes]) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        final_images = []
        
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                # 图片生成进度
                if img := resp.get("streamingImageGenerationResponse"):
                    image_index = img.get("imageIndex", 0)
                    progress = img.get("progress", 0)
                    
                    if self.n == 1 and image_index != self.target_index:
                        continue
                    
                    out_index = 0 if self.n == 1 else image_index
                    
                    yield self._sse("image_generation.partial_image", {
                        "type": "image_generation.partial_image",
                        self.response_field: "",
                        "index": out_index,
                        "progress": progress
                    })
                    continue
                
                # modelResponse
                if mr := resp.get("modelResponse"):
                    if urls := mr.get("generatedImageUrls"):
                        for url in urls:
                            if self.response_format == "url":
                                processed = await self.process_url(url, "image")
                                if processed:
                                    final_images.append(processed)
                                continue
                            dl_service = self._get_dl()
                            base64_data = await dl_service.to_base64(url, self.token, "image")
                            if base64_data:
                                if "," in base64_data:
                                    b64 = base64_data.split(",", 1)[1]
                                else:
                                    b64 = base64_data
                                final_images.append(b64)
                    continue
                    
            for index, b64 in enumerate(final_images):
                if self.n == 1:
                    if index != self.target_index:
                        continue
                    out_index = 0
                else:
                    out_index = index
                
                yield self._sse("image_generation.completed", {
                    "type": "image_generation.completed",
                    self.response_field: b64,
                    "index": out_index,
                    "usage": {
                        "total_tokens": 50,
                        "input_tokens": 25,
                        "output_tokens": 25,
                        "input_tokens_details": {"text_tokens": 5, "image_tokens": 20}
                    }
                })
        except Exception as e:
            logger.error(f"Image stream processing error: {e}")
            raise
        finally:
            await self.close()


class ImageCollectProcessor(BaseProcessor):
    """图片生成非流式响应处理器"""
    
    def __init__(
        self,
        model: str,
        token: str = "",
        response_format: str = "b64_json",
    ):
        super().__init__(model, token)
        self.response_format = (response_format or "b64_json").lower()
    
    async def process(self, response: AsyncIterable[bytes]) -> List[str]:
        """处理并收集图片"""
        images = []
        
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                if mr := resp.get("modelResponse"):
                    if urls := mr.get("generatedImageUrls"):
                        for url in urls:
                            if self.response_format == "url":
                                processed = await self.process_url(url, "image")
                                if processed:
                                    images.append(processed)
                                continue
                            dl_service = self._get_dl()
                            base64_data = await dl_service.to_base64(url, self.token, "image")
                            if base64_data:
                                if "," in base64_data:
                                    b64 = base64_data.split(",", 1)[1]
                                else:
                                    b64 = base64_data
                                images.append(b64)
                                
        except Exception as e:
            logger.error(f"Image collect processing error: {e}")
        finally:
            await self.close()
        
        return images


__all__ = [
    "extract_tool_calls_from_text",
    "StreamProcessor",
    "CollectProcessor",
    "VideoStreamProcessor",
    "VideoCollectProcessor",
    "ImageStreamProcessor",
    "ImageCollectProcessor",
]
