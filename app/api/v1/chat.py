"""
Chat Completions API 路由
"""

from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.core.auth import verify_api_key
from app.services.grok.chat import ChatService
from app.services.grok.model import ModelService
from app.core.exceptions import ValidationException
from app.services.quota import enforce_daily_quota


router = APIRouter(tags=["Chat"])


VALID_ROLES = ["developer", "system", "user", "assistant", "tool"]
USER_CONTENT_TYPES = ["text", "image_url", "input_audio", "file"]
TOOL_CHOICE_TYPES = ["auto", "none", "required"]


class MessageItem(BaseModel):
    """消息项"""
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}")
        return v


class VideoConfig(BaseModel):
    """视频生成配置"""
    aspect_ratio: Optional[str] = Field("3:2", description="视频比例: 3:2, 16:9, 1:1 等")
    video_length: Optional[int] = Field(6, description="视频时长(秒): 5-15")
    resolution: Optional[str] = Field("SD", description="视频分辨率: SD, HD")
    preset: Optional[str] = Field("custom", description="风格预设: fun, normal, spicy")
    
    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v):
        allowed = ["2:3", "3:2", "1:1", "9:16", "16:9"]
        if v and v not in allowed:
            raise ValidationException(
                message=f"aspect_ratio must be one of {allowed}",
                param="video_config.aspect_ratio",
                code="invalid_aspect_ratio"
            )
        return v
    
    @field_validator("video_length")
    @classmethod
    def validate_video_length(cls, v):
        if v is not None:
            if v < 5 or v > 15:
                raise ValidationException(
                    message="video_length must be between 5 and 15 seconds",
                    param="video_config.video_length",
                    code="invalid_video_length"
                )
        return v

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v):
        allowed = ["SD", "HD"]
        if v and v not in allowed:
            raise ValidationException(
                message=f"resolution must be one of {allowed}",
                param="video_config.resolution",
                code="invalid_resolution"
            )
        return v
    
    @field_validator("preset")
    @classmethod
    def validate_preset(cls, v):
        # 允许为空，默认 custom
        if not v:
            return "custom"
        allowed = ["fun", "normal", "spicy", "custom"]
        if v not in allowed:
             raise ValidationException(
                message=f"preset must be one of {allowed}",
                param="video_config.preset",
                code="invalid_preset"
             )
        return v


class ChatCompletionRequest(BaseModel):
    """Chat Completions 请求"""
    model: str = Field(..., description="模型名称")
    messages: List[MessageItem] = Field(..., description="消息数组")
    stream: Optional[bool] = Field(None, description="是否流式输出")
    thinking: Optional[str] = Field(None, description="思考模式: enabled/disabled/None")
    response_format: Optional[Union[str, Dict[str, Any]]] = Field(None, description="响应格式约束")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="工具定义列表")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="工具选择策略")
    parallel_tool_calls: Optional[bool] = Field(None, description="是否允许并行工具调用")
    
    # 视频生成配置
    video_config: Optional[VideoConfig] = Field(None, description="视频生成参数")
    
    model_config = {
        "extra": "ignore"
    }


def validate_request(request: ChatCompletionRequest):
    """验证请求参数"""
    # 验证模型
    if not ModelService.valid(request.model):
        raise ValidationException(
            message=f"The model `{request.model}` does not exist or you do not have access to it.",
            param="model",
            code="model_not_found"
        )

    # 验证 tools
    tool_names = set()
    if request.tools is not None:
        if not isinstance(request.tools, list):
            raise ValidationException(
                message="tools must be an array",
                param="tools",
                code="invalid_tools"
            )

        for tool_idx, tool in enumerate(request.tools):
            if not isinstance(tool, dict):
                raise ValidationException(
                    message="Each tool must be an object",
                    param=f"tools.{tool_idx}",
                    code="invalid_tools"
                )

            if tool.get("type") != "function":
                raise ValidationException(
                    message="Only function tools are supported",
                    param=f"tools.{tool_idx}.type",
                    code="invalid_tools"
                )

            function = tool.get("function")
            if not isinstance(function, dict):
                raise ValidationException(
                    message="Tool must include function object",
                    param=f"tools.{tool_idx}.function",
                    code="invalid_tools"
                )

            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValidationException(
                    message="Tool function name cannot be empty",
                    param=f"tools.{tool_idx}.function.name",
                    code="invalid_tools"
                )

            clean_name = name.strip()
            if clean_name in tool_names:
                raise ValidationException(
                    message=f"Duplicate tool name: {clean_name}",
                    param=f"tools.{tool_idx}.function.name",
                    code="invalid_tools"
                )
            tool_names.add(clean_name)

            parameters = function.get("parameters")
            if parameters is not None and not isinstance(parameters, dict):
                raise ValidationException(
                    message="Tool function parameters must be an object",
                    param=f"tools.{tool_idx}.function.parameters",
                    code="invalid_tools"
                )

    # 验证 tool_choice
    tool_choice = request.tool_choice
    if tool_choice is not None:
        if isinstance(tool_choice, str):
            if tool_choice not in TOOL_CHOICE_TYPES:
                raise ValidationException(
                    message=f"tool_choice must be one of {TOOL_CHOICE_TYPES}",
                    param="tool_choice",
                    code="invalid_tool_choice"
                )
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") != "function":
                raise ValidationException(
                    message="tool_choice.type must be 'function'",
                    param="tool_choice.type",
                    code="invalid_tool_choice"
                )

            function = tool_choice.get("function")
            if not isinstance(function, dict):
                raise ValidationException(
                    message="tool_choice.function must be an object",
                    param="tool_choice.function",
                    code="invalid_tool_choice"
                )

            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValidationException(
                    message="tool_choice.function.name cannot be empty",
                    param="tool_choice.function.name",
                    code="invalid_tool_choice"
                )

            if request.tools is None:
                raise ValidationException(
                    message="tool_choice requires tools",
                    param="tool_choice",
                    code="invalid_tool_choice"
                )

            if tool_names and name.strip() not in tool_names:
                raise ValidationException(
                    message=f"tool_choice function '{name.strip()}' was not found in tools",
                    param="tool_choice.function.name",
                    code="invalid_tool_choice"
                )
        else:
            raise ValidationException(
                message="tool_choice must be a string or object",
                param="tool_choice",
                code="invalid_tool_choice"
            )

    # Grok 上游不支持原生 response_format，服务端仅在内容层注入提示词约束。
    # 因此这里不做严格枚举校验，只做基本类型检查（string/object）。
    response_format = request.response_format
    if response_format is not None and not isinstance(response_format, (str, dict)):
        raise ValidationException(
            message="response_format must be a string or object",
            param="response_format",
            code="invalid_response_format"
        )
    
    # 验证消息
    for idx, msg in enumerate(request.messages):
        content = msg.content
        role = msg.role

        if msg.tool_call_id is not None:
            if role != "tool":
                raise ValidationException(
                    message="tool_call_id is only valid for role=tool",
                    param=f"messages.{idx}.tool_call_id",
                    code="invalid_tool_message"
                )
            if not isinstance(msg.tool_call_id, str) or not msg.tool_call_id.strip():
                raise ValidationException(
                    message="tool_call_id cannot be empty",
                    param=f"messages.{idx}.tool_call_id",
                    code="invalid_tool_message"
                )

        if msg.tool_calls is not None:
            if role != "assistant":
                raise ValidationException(
                    message="tool_calls are only valid for role=assistant",
                    param=f"messages.{idx}.tool_calls",
                    code="invalid_tool_message"
                )
            if not isinstance(msg.tool_calls, list) or not msg.tool_calls:
                raise ValidationException(
                    message="tool_calls must be a non-empty array",
                    param=f"messages.{idx}.tool_calls",
                    code="invalid_tool_message"
                )

            for call_idx, call in enumerate(msg.tool_calls):
                if not isinstance(call, dict):
                    raise ValidationException(
                        message="Each tool_call must be an object",
                        param=f"messages.{idx}.tool_calls.{call_idx}",
                        code="invalid_tool_message"
                    )

                call_type = call.get("type", "function")
                if call_type != "function":
                    raise ValidationException(
                        message="tool_calls only support type='function'",
                        param=f"messages.{idx}.tool_calls.{call_idx}.type",
                        code="invalid_tool_message"
                    )

                call_id = call.get("id", "")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValidationException(
                        message="tool_call id cannot be empty",
                        param=f"messages.{idx}.tool_calls.{call_idx}.id",
                        code="invalid_tool_message"
                    )

                function = call.get("function")
                if not isinstance(function, dict):
                    raise ValidationException(
                        message="tool_call.function must be an object",
                        param=f"messages.{idx}.tool_calls.{call_idx}.function",
                        code="invalid_tool_message"
                    )

                name = function.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise ValidationException(
                        message="tool_call.function.name cannot be empty",
                        param=f"messages.{idx}.tool_calls.{call_idx}.function.name",
                        code="invalid_tool_message"
                    )

                if tool_names and name.strip() not in tool_names:
                    raise ValidationException(
                        message=f"tool_call.function.name '{name.strip()}' was not found in tools",
                        param=f"messages.{idx}.tool_calls.{call_idx}.function.name",
                        code="invalid_tool_message"
                    )

                arguments = function.get("arguments")
                if arguments is not None and not isinstance(arguments, (str, dict, list, int, float, bool)):
                    raise ValidationException(
                        message="tool_call.function.arguments must be string or JSON value",
                        param=f"messages.{idx}.tool_calls.{call_idx}.function.arguments",
                        code="invalid_tool_message"
                    )

        # OpenClaw 兼容：assistant/tool 允许 content 为 null。
        if content is None:
            if role in ("assistant", "tool"):
                continue
            raise ValidationException(
                message="Message content cannot be empty",
                param=f"messages.{idx}.content",
                code="empty_content"
            )

        # 字符串内容
        if isinstance(content, str):
            if not content.strip():
                raise ValidationException(
                    message="Message content cannot be empty",
                    param=f"messages.{idx}.content",
                    code="empty_content"
                )
        
        # 列表内容
        elif isinstance(content, list):
            if not content:
                raise ValidationException(
                    message="Message content cannot be an empty array",
                    param=f"messages.{idx}.content",
                    code="empty_content"
                )
            
            for block_idx, block in enumerate(content):
                # 检查空对象
                if not block:
                    raise ValidationException(
                        message="Content block cannot be empty",
                        param=f"messages.{idx}.content.{block_idx}",
                        code="empty_block"
                    )
                
                # 检查 type 字段
                if "type" not in block:
                    raise ValidationException(
                        message="Content block must have a 'type' field",
                        param=f"messages.{idx}.content.{block_idx}",
                        code="missing_type"
                    )
                
                block_type = block.get("type")
                
                # 检查 type 空值
                if not block_type or not isinstance(block_type, str) or not block_type.strip():
                    raise ValidationException(
                        message="Content block 'type' cannot be empty",
                        param=f"messages.{idx}.content.{block_idx}.type",
                        code="empty_type"
                    )
                
                # 验证 type 有效性
                if msg.role == "user":
                    if block_type not in USER_CONTENT_TYPES:
                        raise ValidationException(
                            message=f"Invalid content block type: '{block_type}'",
                            param=f"messages.{idx}.content.{block_idx}.type",
                            code="invalid_type"
                        )
                elif block_type != "text":
                    raise ValidationException(
                        message=f"The `{msg.role}` role only supports 'text' type, got '{block_type}'",
                        param=f"messages.{idx}.content.{block_idx}.type",
                        code="invalid_type"
                    )
                
                # 验证字段是否存在 & 非空
                if block_type == "text":
                    text = block.get("text", "")
                    if not isinstance(text, str) or not text.strip():
                        raise ValidationException(
                            message="Text content cannot be empty",
                            param=f"messages.{idx}.content.{block_idx}.text",
                            code="empty_text"
                        )
                elif block_type == "image_url":
                    image_url = block.get("image_url")
                    if not image_url or not (isinstance(image_url, dict) and image_url.get("url")):
                        raise ValidationException(
                            message="image_url must have a 'url' field",
                            param=f"messages.{idx}.content.{block_idx}.image_url",
                            code="missing_url"
                        )


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: Optional[str] = Depends(verify_api_key)):
    """Chat Completions API - 兼容 OpenAI"""
    
    # 参数验证
    validate_request(request)

    # Daily quota (best-effort)
    await enforce_daily_quota(api_key, request.model)
    
    # 检测视频模型
    model_info = ModelService.get(request.model)
    if model_info and model_info.is_video:
        from app.services.grok.media import VideoService
        
        # 提取视频配置 (默认值在 Pydantic 模型中处理)
        v_conf = request.video_config or VideoConfig()
        
        result = await VideoService.completions(
            model=request.model,
            messages=[msg.model_dump() for msg in request.messages],
            stream=request.stream,
            thinking=request.thinking,
            aspect_ratio=v_conf.aspect_ratio,
            video_length=v_conf.video_length,
            resolution=v_conf.resolution,
            preset=v_conf.preset
        )
    else:
        result = await ChatService.completions(
            model=request.model,
            messages=[msg.model_dump() for msg in request.messages],
            stream=request.stream,
            thinking=request.thinking,
            response_format=request.response_format,
            tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls
        )
    
    if isinstance(result, dict):
        return JSONResponse(content=result)
    else:
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )


__all__ = ["router"]
