"""阿里云百炼 TTS 语音发送插件。"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import asyncio
import base64
import json
import logging
import random
import wave

from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType
from maibot_sdk import Command, Field, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import ToolParameterInfo, ToolParamType


DEFAULT_TTS_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
DEFAULT_REALTIME_ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DEFAULT_MODEL = "qwen3-tts-instruct-flash-realtime"
DEFAULT_VOICE = "Sunny"
DEFAULT_FORMAT = "wav"
DEFAULT_SAMPLE_RATE = 24000
MAX_AUDIO_BYTES = 12 * 1024 * 1024
SUPPORTED_EMOTIONS = {"neutral", "fearful", "angry", "sad", "surprised", "happy", "disgusted"}
SUPPORTED_SCENES = {
    "闲聊对话",
    "闲聊互动",
    "自由对话",
    "比赛解说",
    "深夜电台广播",
    "剧情解说",
    "诗歌朗诵",
    "科普知识推广",
    "产品推广",
    "脱口秀表演",
    "广告促销",
}
CHILDLIKE_VOICES = {"longhuhu_v3", "longpaopao_v3", "longxian_v3", "longling_v3", "longshanshan_v3", "longniuniu_v3"}
QWEN_TTS_MODELS = (
    "qwen-tts",
    "qwen3-tts",
)

logger = logging.getLogger("plugin.aliyun_tts_plugin")


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用插件")
    config_version: str = Field(default="1.0.0", description="配置版本")


class AliyunTTSConfig(PluginConfigBase):
    """阿里云百炼 TTS 接口配置。"""

    __ui_label__ = "百炼 TTS"
    __ui_icon__ = "volume-2"
    __ui_order__ = 1

    api_key: str = Field(default="", description="阿里云百炼 DashScope API Key")
    endpoint: str = Field(default=DEFAULT_TTS_ENDPOINT, description="语音合成接口地址")
    realtime_endpoint: str = Field(default=DEFAULT_REALTIME_ENDPOINT, description="实时语音合成 WebSocket 地址")
    use_realtime: bool = Field(default=True, description="是否优先使用 Qwen 实时语音合成")
    model: str = Field(default=DEFAULT_MODEL, description="语音合成模型")
    voice: str = Field(default=DEFAULT_VOICE, description="默认音色")
    audio_format: str = Field(default=DEFAULT_FORMAT, description="音频格式：wav、mp3、opus 或 pcm")
    sample_rate: int = Field(default=DEFAULT_SAMPLE_RATE, description="采样率：8000、16000、22050、24000、44100、48000")
    volume: int = Field(default=35, description="音量，范围 0-100")
    rate: float = Field(default=1.0, description="语速，范围 0.5-2.0")
    pitch: float = Field(default=1.0, description="音高，范围 0.5-2.0")
    timeout_seconds: int = Field(default=60, description="请求超时时间，单位秒")
    max_text_length: int = Field(default=300, description="单次合成文本最大长度")
    enable_markdown_filter: bool = Field(default=True, description="是否请求百炼过滤 Markdown 标记")
    enable_aigc_tag: bool = Field(default=False, description="是否在音频中添加 AIGC 隐性标识")
    enable_auto_instruction: bool = Field(default=True, description="未手动指定 instruction 时，是否根据文本自动设置场景和情绪")
    default_scene: str = Field(default="闲聊对话", description="默认场景")
    default_emotion: str = Field(default="happy", description="默认情绪")
    enable_noise: bool = Field(default=True, description="是否给 wav 语音添加轻微底噪")
    noise_level: float = Field(default=0.003, description="底噪强度，建议 0-0.02")


class ToolSwitchConfig(PluginConfigBase):
    """工具启用配置。"""

    __ui_label__ = "工具"
    __ui_icon__ = "wrench"
    __ui_order__ = 2

    enable_send_tts_voice: bool = Field(default=True, description="是否启用发送 TTS 语音工具")
    enable_tts_test_command: bool = Field(default=True, description="是否启用 TTS 测试命令")


class AliyunTTSPluginConfig(PluginConfigBase):
    """阿里云百炼 TTS 插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    tts: AliyunTTSConfig = Field(default_factory=AliyunTTSConfig)
    tools: ToolSwitchConfig = Field(default_factory=ToolSwitchConfig)


class AliyunTTSPlugin(MaiBotPlugin):
    """通过阿里云百炼 CosyVoice 生成并发送语音。"""

    config_model = AliyunTTSPluginConfig

    async def on_load(self) -> None:
        """插件加载时记录当前状态。"""

        logger.info("阿里云百炼 TTS 插件已加载")

    async def on_unload(self) -> None:
        """插件卸载时执行清理。"""

    async def on_config_update(self, scope: str, config_data: Dict[str, object], version: str) -> None:
        """处理配置热更新。"""

        del scope
        del config_data
        del version

    def _validate_text(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """校验并规范化待合成文本。"""

        normalized_text = str(text or "").strip()
        if not normalized_text:
            return None, "缺少要合成的文本"

        max_text_length = self.config.tts.max_text_length
        if max_text_length > 0 and len(normalized_text) > max_text_length:
            return None, f"文本过长，当前最多允许 {max_text_length} 个字符"
        return normalized_text, None

    @staticmethod
    def _normalize_emotion(emotion: str) -> str:
        """规范化 Instruct 情绪值。"""

        normalized_emotion = str(emotion or "").strip().lower()
        if normalized_emotion in SUPPORTED_EMOTIONS:
            return normalized_emotion
        return ""

    @staticmethod
    def _normalize_scene(scene: str) -> str:
        """规范化 Instruct 场景值。"""

        normalized_scene = str(scene or "").strip()
        if normalized_scene in SUPPORTED_SCENES:
            return normalized_scene
        return ""

    def _normalize_voice(self, voice: str) -> str:
        """规范化音色，兼容模型传入的 default/auto 等占位值。"""

        normalized_voice = str(voice or "").strip()
        if normalized_voice.lower() in {"", "default", "auto", "none", "null"}:
            normalized_voice = self.config.tts.voice.strip()
        if normalized_voice in CHILDLIKE_VOICES:
            normalized_voice = DEFAULT_VOICE
        return normalized_voice

    def _uses_qwen_tts(self) -> bool:
        """判断当前配置是否使用千问 TTS 接口。"""

        normalized_model = self.config.tts.model.strip().lower()
        return normalized_model.startswith(QWEN_TTS_MODELS)

    def _uses_realtime_tts(self) -> bool:
        """判断当前配置是否使用千问实时 TTS 接口。"""

        normalized_model = self.config.tts.model.strip().lower()
        return self.config.tts.use_realtime and normalized_model.startswith(QWEN_TTS_MODELS) and "realtime" in normalized_model

    @staticmethod
    def _is_supported_instruction(instruction: str, voice: str) -> bool:
        """判断 instruction 是否已经符合百炼 Instruct 的固定格式。"""

        normalized_instruction = str(instruction or "").strip()
        if not normalized_instruction:
            return False
        if not normalized_instruction.endswith("。"):
            return False
        if "你说话的情感是" in normalized_instruction:
            return True
        return False

    def _infer_emotion_from_text(self, text: str) -> str:
        """根据文本内容粗略推断适合的情绪。"""

        normalized_text = str(text or "")
        if any(keyword in normalized_text for keyword in ("哈哈", "开心", "高兴", "好耶", "太棒", "喜欢", "可爱", "调侃")):
            return "happy"
        if any(keyword in normalized_text for keyword in ("！", "惊", "居然", "竟然", "真的吗", "不会吧")):
            return "surprised"
        if any(keyword in normalized_text for keyword in ("难过", "伤心", "哭", "遗憾", "对不起", "抱歉", "失落")):
            return "sad"
        if any(keyword in normalized_text for keyword in ("生气", "气死", "讨厌", "过分", "离谱", "不许")):
            return "angry"
        if any(keyword in normalized_text for keyword in ("害怕", "吓", "可怕", "恐怖", "担心", "紧张")):
            return "fearful"
        if any(keyword in normalized_text for keyword in ("恶心", "嫌弃", "反感", "下头", "不屑")):
            return "disgusted"
        return self._normalize_emotion(self.config.tts.default_emotion) or "neutral"

    def _infer_scene_from_text(self, text: str) -> str:
        """根据文本内容粗略推断适合的场景。"""

        normalized_text = str(text or "")
        if any(keyword in normalized_text for keyword in ("诗", "月光", "朗诵", "远方", "春风")):
            return "诗歌朗诵"
        if any(keyword in normalized_text for keyword in ("比赛", "选手", "得分", "进球", "冠军")):
            return "比赛解说"
        if any(keyword in normalized_text for keyword in ("深夜", "电台", "晚安", "睡前")):
            return "深夜电台广播"
        if any(keyword in normalized_text for keyword in ("剧情", "故事", "角色", "旁白", "然后")):
            return "剧情解说"
        if any(keyword in normalized_text for keyword in ("科普", "知识", "原理", "为什么", "实验")):
            return "科普知识推广"
        if any(keyword in normalized_text for keyword in ("推荐", "优惠", "购买", "产品", "限时")):
            return "产品推广"
        if any(keyword in normalized_text for keyword in ("段子", "吐槽", "笑话", "脱口秀")):
            return "脱口秀表演"
        return self._normalize_scene(self.config.tts.default_scene) or "闲聊对话"

    def _build_instruction(self, text: str, instruction: str, emotion: str, scene: str, voice: str) -> str:
        """生成符合百炼 Instruct 格式的指令。"""

        normalized_instruction = str(instruction or "").strip()
        if self._uses_qwen_tts():
            if normalized_instruction:
                return normalized_instruction
            emotion_hint = self._normalize_emotion(emotion) or self._infer_emotion_from_text(text)
            scene_hint = self._normalize_scene(scene) or self._infer_scene_from_text(text)
            return (
                f"使用{voice}的撒娇搞怪、活泼可爱的少女感音色，"
                f"语气自然亲近，情绪偏{emotion_hint}，适合{scene_hint}，音量稍小。"
            )

        if self._is_supported_instruction(normalized_instruction, voice):
            return normalized_instruction
        instruction_hint = "" if not normalized_instruction else f" {normalized_instruction}"

        normalized_emotion = self._normalize_emotion(emotion)
        normalized_scene = self._normalize_scene(scene)
        if not self.config.tts.enable_auto_instruction:
            normalized_emotion = normalized_emotion or self._normalize_emotion(self.config.tts.default_emotion) or "neutral"
            normalized_scene = normalized_scene or self._normalize_scene(self.config.tts.default_scene) or "闲聊对话"
            return f"你正在进行{normalized_scene}，你说话的情感是{normalized_emotion}。"

        inferred_emotion = normalized_emotion or self._infer_emotion_from_text(f"{text}{instruction_hint}")
        inferred_scene = normalized_scene or self._infer_scene_from_text(text)
        return f"你正在进行{inferred_scene}，你说话的情感是{inferred_emotion}。"

    def _build_tts_payload(self, text: str, voice: str, instruction: str) -> Dict[str, Any]:
        """构造百炼语音合成请求体。"""

        if self._uses_qwen_tts():
            payload: Dict[str, Any] = {
                "model": self.config.tts.model,
                "input": {
                    "text": text,
                    "voice": voice,
                    "language_type": "Chinese",
                },
            }
            normalized_instruction = str(instruction or "").strip()
            if normalized_instruction:
                payload["parameters"] = {
                    "instructions": normalized_instruction,
                    "optimize_instructions": True,
                }
            return payload

        input_payload: Dict[str, Any] = {
            "text": text,
            "voice": voice,
            "format": self.config.tts.audio_format,
            "sample_rate": self.config.tts.sample_rate,
            "volume": self.config.tts.volume,
            "rate": self.config.tts.rate,
            "pitch": self.config.tts.pitch,
            "enable_markdown_filter": self.config.tts.enable_markdown_filter,
            "enable_aigc_tag": self.config.tts.enable_aigc_tag,
        }
        normalized_instruction = str(instruction or "").strip()
        if normalized_instruction:
            input_payload["instruction"] = normalized_instruction

        return {
            "model": self.config.tts.model,
            "input": input_payload,
        }

    @staticmethod
    def _extract_audio_url(payload: Dict[str, Any]) -> str:
        """从百炼返回体中提取完整音频 URL。"""

        output = payload.get("output")
        if not isinstance(output, dict):
            return ""
        audio = output.get("audio")
        if not isinstance(audio, dict):
            return ""
        return str(audio.get("url") or "").strip()

    @staticmethod
    def _extract_error_message(payload: Any) -> str:
        """从错误响应中提取可读错误信息。"""

        if not isinstance(payload, dict):
            return ""
        for key in ("message", "error", "code"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _new_event(event_type: str, **payload: Any) -> Dict[str, Any]:
        """构造 Qwen Realtime WebSocket 事件。"""

        return {
            "event_id": f"event_{uuid4().hex}",
            "type": event_type,
            **payload,
        }

    def _build_realtime_ws_url(self) -> str:
        """构造带模型参数的实时 TTS WebSocket 地址。"""

        endpoint = self.config.tts.realtime_endpoint.strip() or DEFAULT_REALTIME_ENDPOINT
        if "model=" in endpoint:
            return endpoint
        separator = "&" if "?" in endpoint else "?"
        return f"{endpoint}{separator}model={self.config.tts.model.strip()}"

    def _build_realtime_session(self, voice: str, instruction: str) -> Dict[str, Any]:
        """构造实时 TTS 会话配置。"""

        session_payload: Dict[str, Any] = {
            "voice": voice,
            "mode": "server_commit",
            "language_type": "Chinese",
            "response_format": "pcm",
            "sample_rate": self.config.tts.sample_rate,
            "speech_rate": self.config.tts.rate,
            "volume": self.config.tts.volume,
            "pitch_rate": self.config.tts.pitch,
        }
        normalized_instruction = str(instruction or "").strip()
        if normalized_instruction:
            session_payload["instructions"] = normalized_instruction
            session_payload["optimize_instructions"] = True
        return session_payload

    async def _request_realtime_audio(self, text: str, voice: str, instruction: str) -> bytes:
        """调用 Qwen 实时 TTS，收集 PCM 音频并封装为 wav。"""

        api_key = self.config.tts.api_key.strip()
        if not api_key:
            raise RuntimeError("未配置阿里云百炼 API Key")

        audio_chunks: list[bytes] = []
        headers = {"Authorization": f"Bearer {api_key}"}
        timeout = ClientTimeout(total=max(self.config.tts.timeout_seconds, 1))
        async with ClientSession(timeout=timeout) as session:
            async with session.ws_connect(self._build_realtime_ws_url(), headers=headers) as ws:
                await ws.send_str(
                    json.dumps(
                        self._new_event("session.update", session=self._build_realtime_session(voice, instruction)),
                        ensure_ascii=False,
                    )
                )
                await ws.send_str(
                    json.dumps(self._new_event("input_text_buffer.append", text=text), ensure_ascii=False)
                )
                await ws.send_str(json.dumps(self._new_event("session.finish"), ensure_ascii=False))

                async for message in ws:
                    if message.type == WSMsgType.TEXT:
                        event = json.loads(message.data)
                        event_type = str(event.get("type") or "")
                        if event_type == "error":
                            error_payload = event.get("error")
                            error_message = self._extract_error_message(error_payload)
                            raise RuntimeError(error_message or str(error_payload or event))
                        if event_type == "response.audio.delta":
                            audio_delta = str(event.get("delta") or "")
                            if audio_delta:
                                audio_chunks.append(base64.b64decode(audio_delta))
                            continue
                        if event_type in {"response.done", "session.finished"}:
                            break
                        continue
                    if message.type == WSMsgType.ERROR:
                        raise RuntimeError(f"Qwen 实时 TTS WebSocket 异常：{ws.exception()}")
                    if message.type in {WSMsgType.CLOSED, WSMsgType.CLOSE}:
                        break

        if not audio_chunks:
            raise RuntimeError("Qwen 实时 TTS 未返回音频数据")
        audio_bytes = self._build_wav_from_pcm(b"".join(audio_chunks), self.config.tts.sample_rate)
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise RuntimeError("TTS 音频过大，无法发送")
        return self._apply_audio_effects(audio_bytes)

    @staticmethod
    def _build_wav_from_pcm(pcm_bytes: bytes, sample_rate: int) -> bytes:
        """将实时 TTS 返回的 16-bit mono PCM 封装为 wav。"""

        output_buffer = BytesIO()
        with wave.open(output_buffer, "wb") as output_wav:
            output_wav.setnchannels(1)
            output_wav.setsampwidth(2)
            output_wav.setframerate(sample_rate)
            output_wav.writeframes(pcm_bytes)
        return output_buffer.getvalue()

    async def _request_tts(self, text: str, voice: str, instruction: str) -> str:
        """调用百炼非流式 TTS 接口，返回可下载的音频 URL。"""

        api_key = self.config.tts.api_key.strip()
        if not api_key:
            raise RuntimeError("未配置阿里云百炼 API Key")

        endpoint = self.config.tts.endpoint.strip() or DEFAULT_TTS_ENDPOINT
        payload = self._build_tts_payload(text, voice, instruction)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        timeout = ClientTimeout(total=max(self.config.tts.timeout_seconds, 1))

        async with ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                response_payload = await response.json(content_type=None)
                if response.status >= 400:
                    error_message = self._extract_error_message(response_payload)
                    raise RuntimeError(error_message or f"百炼 TTS 请求失败，HTTP {response.status}")

        audio_url = self._extract_audio_url(response_payload)
        if not audio_url:
            error_message = self._extract_error_message(response_payload)
            raise RuntimeError(error_message or "百炼 TTS 返回中缺少音频 URL")
        return audio_url

    async def _download_audio_base64(self, audio_url: str) -> str:
        """下载百炼生成的音频并转换为 Base64。"""

        timeout = ClientTimeout(total=max(self.config.tts.timeout_seconds, 1))
        async with ClientSession(timeout=timeout) as session:
            async with session.get(audio_url) as response:
                if response.status >= 400:
                    raise RuntimeError(f"下载 TTS 音频失败，HTTP {response.status}")

                audio_bytes = await response.read()
        if not audio_bytes:
            raise RuntimeError("下载到的 TTS 音频为空")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise RuntimeError("TTS 音频过大，无法发送")
        audio_bytes = self._apply_audio_effects(audio_bytes)
        return base64.b64encode(audio_bytes).decode("ascii")

    def _apply_audio_effects(self, audio_bytes: bytes) -> bytes:
        """对 wav 音频做轻量后处理，目前用于添加一点自然底噪。"""

        if not self.config.tts.enable_noise:
            return audio_bytes
        if self.config.tts.audio_format.strip().lower() != "wav":
            return audio_bytes

        noise_level = max(0.0, min(float(self.config.tts.noise_level), 0.02))
        if noise_level <= 0:
            return audio_bytes

        try:
            with wave.open(BytesIO(audio_bytes), "rb") as source_wav:
                params = source_wav.getparams()
                sample_width = source_wav.getsampwidth()
                frame_count = source_wav.getnframes()
                raw_frames = source_wav.readframes(frame_count)
        except wave.Error:
            return audio_bytes

        if sample_width != 2:
            return audio_bytes

        max_sample = 32767
        noise_amplitude = int(max_sample * noise_level)
        processed = bytearray(raw_frames)
        for index in range(0, len(processed) - 1, 2):
            sample = int.from_bytes(processed[index : index + 2], byteorder="little", signed=True)
            sample = max(-32768, min(32767, sample + random.randint(-noise_amplitude, noise_amplitude)))
            processed[index : index + 2] = int(sample).to_bytes(2, byteorder="little", signed=True)

        output_buffer = BytesIO()
        with wave.open(output_buffer, "wb") as output_wav:
            output_wav.setparams(params)
            output_wav.writeframes(bytes(processed))
        return output_buffer.getvalue()

    async def _send_voice(self, stream_id: str, audio_base64: str) -> bool:
        """通过 Host 自定义消息能力发送语音组件。"""

        return bool(
            await self.ctx.send.custom(
                "voice",
                audio_base64,
                stream_id,
                sync_to_maisaka_history=True,
                maisaka_source_kind="plugin_send",
            )
        )

    @Tool(
        "send_tts_voice",
        description=(
            "把指定文本合成为语音并发送到当前会话。"
            "适合在需要用语音回复、朗读短句、或用户明确要求发语音时使用。"
        ),
        parameters=[
            ToolParameterInfo(
                name="text",
                param_type=ToolParamType.STRING,
                description="要合成为语音并发送的文本",
                required=True,
            ),
            ToolParameterInfo(
                name="voice",
                param_type=ToolParamType.STRING,
                description="可选音色，不填则使用插件配置中的默认音色",
                required=False,
            ),
            ToolParameterInfo(
                name="instruction",
                param_type=ToolParamType.STRING,
                description="可选百炼 Instruct 固定格式指令，例如：你正在进行闲聊对话，你说话的情感是happy。",
                required=False,
            ),
            ToolParameterInfo(
                name="emotion",
                param_type=ToolParamType.STRING,
                description="可选情绪：neutral、fearful、angry、sad、surprised、happy、disgusted",
                required=False,
            ),
            ToolParameterInfo(
                name="scene",
                param_type=ToolParamType.STRING,
                description="可选场景：闲聊对话、比赛解说、深夜电台广播、剧情解说、诗歌朗诵、科普知识推广、产品推广、脱口秀表演",
                required=False,
            ),
        ],
    )
    async def handle_send_tts_voice(
        self,
        text: str = "",
        voice: str = "",
        instruction: str = "",
        emotion: str = "",
        scene: str = "",
        stream_id: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """生成并发送 TTS 语音。"""

        del kwargs

        if not self.config.plugin.enabled:
            return {"success": False, "content": "阿里云百炼 TTS 插件已禁用。"}
        if not self.config.tools.enable_send_tts_voice:
            return {"success": False, "content": "发送 TTS 语音工具未启用。"}
        if not stream_id:
            return {"success": False, "content": "缺少当前会话 stream_id，无法发送语音。"}

        normalized_text, text_error = self._validate_text(text)
        if normalized_text is None:
            return {"success": False, "content": text_error or "文本无效。"}

        normalized_voice = self._normalize_voice(voice)
        if not normalized_voice:
            return {"success": False, "content": "缺少音色配置。"}

        try:
            effective_instruction = self._build_instruction(normalized_text, instruction, emotion, scene, normalized_voice)
            audio_url = ""
            if self._uses_realtime_tts():
                audio_bytes = await self._request_realtime_audio(normalized_text, normalized_voice, effective_instruction)
                audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
            else:
                audio_url = await self._request_tts(normalized_text, normalized_voice, effective_instruction)
                audio_base64 = await self._download_audio_base64(audio_url)
            sent = await self._send_voice(stream_id, audio_base64)
        except (ClientError, TimeoutError) as exc:
            logger.warning("调用阿里云百炼 TTS 失败: %s", exc)
            return {"success": False, "content": f"调用阿里云百炼 TTS 失败：{exc}"}
        except Exception as exc:
            logger.warning("发送 TTS 语音失败: %s", exc)
            return {"success": False, "content": f"发送 TTS 语音失败：{exc}"}

        if not sent:
            return {"success": False, "content": "语音已合成，但发送失败。"}
        return {
            "success": True,
            "content": "已发送 TTS 语音。",
            "voice": normalized_voice,
            "instruction": effective_instruction,
            "audio_url": audio_url,
        }

    @Command(
        "tts_test",
        description="快速测试阿里云百炼 TTS 语音发送",
        pattern=r"^/(?:tts|tts_test|语音测试)(?:\s+(?P<text>.+))?$",
    )
    async def handle_tts_test_command(
        self,
        stream_id: str = "",
        matched_groups: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[bool, str, bool]:
        """通过命令快速测试 TTS 语音发送。"""

        del kwargs

        if not self.config.plugin.enabled:
            return False, "阿里云百炼 TTS 插件已禁用。", True
        if not self.config.tools.enable_tts_test_command:
            return False, "TTS 测试命令未启用。", True
        if not stream_id:
            return False, "缺少当前会话 stream_id，无法发送语音。", True

        command_text = ""
        if isinstance(matched_groups, dict):
            command_text = str(matched_groups.get("text") or "").strip()
        if not command_text:
            command_text = "这是一条来自阿里云百炼的语音测试。"

        asyncio.create_task(self._run_tts_test_command(command_text, stream_id), name="aliyun-tts-test-command")
        return True, "已开始合成 TTS 测试语音，稍后发送。", True

    async def _run_tts_test_command(self, text: str, stream_id: str) -> None:
        """后台执行测试命令，避免插件命令 30 秒超时。"""

        result = await self.handle_send_tts_voice(text=text, stream_id=stream_id)
        if result.get("success"):
            return

        content = str(result.get("content") or "TTS 测试语音发送失败。")
        logger.warning("TTS 测试命令后台执行失败: %s", content)
        try:
            await self.ctx.send.text(content, stream_id)
        except Exception as exc:
            logger.warning("发送 TTS 测试失败提示失败: %s", exc)


def create_plugin() -> AliyunTTSPlugin:
    """创建插件实例。"""

    return AliyunTTSPlugin()
