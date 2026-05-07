# 阿里云百炼 TTS 插件

阿里云百炼 TTS 插件用于在 MaiBot 中调用 DashScope / 百炼语音合成能力，将文本合成为语音并发送到当前会话。

插件默认优先使用 Qwen 实时语音合成接口，也保留普通语音合成接口配置。支持自动生成朗读指令、过滤 Markdown 标记、限制单次文本长度，并可为 wav 音频添加轻微底噪。

## 功能

- 通过 `send_tts_voice` 工具发送 TTS 语音。
- 支持 `/tts_test` 测试命令。
- 支持 Qwen 实时语音合成和普通 HTTP 语音合成接口。
- 支持场景、情绪、音色、语速、音量和音高配置。
- 支持 Markdown 过滤和 AIGC 标识开关。

## 配置

```toml
[plugin]
enabled = true
config_version = "1.0.0"

[tts]
api_key = ""
endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
realtime_endpoint = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
use_realtime = true
model = "qwen3-tts-instruct-flash-realtime"
voice = "Sunny"
audio_format = "wav"
sample_rate = 24000
volume = 35
rate = 1.0
pitch = 1.0
timeout_seconds = 60
max_text_length = 300
enable_markdown_filter = true
enable_aigc_tag = false
enable_auto_instruction = true
default_scene = "闲聊对话"
default_emotion = "happy"
enable_noise = true
noise_level = 0.003

[tools]
enable_send_tts_voice = true
enable_tts_test_command = true
```

`api_key` 请填写阿里云百炼 DashScope API Key。不要将真实 Key 提交到公开仓库。

## 命令和工具

- `/tts_test`：发送一段测试语音。
- `send_tts_voice`：供 LLM 调用，将指定文本合成为语音发送。

## 许可证

MIT
