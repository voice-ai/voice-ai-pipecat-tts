# Voice.AI TTS Service for Pipecat

An official integration for [Voice.AI](https://voice.ai) text-to-speech (TTS) with [Pipecat](https://github.com/pipecat-ai/pipecat).

## Overview

This integration provides streaming text-to-speech capabilities using Voice.AI's WebSocket API. 

## Features

- Streaming text-to-speech via WebSocket API
- Support for 11 languages (English, Catalan, Swedish, Spanish, French, German, Italian, Portuguese, Polish, Russian, Dutch)
- Custom voice cloning support

## Installation

### Prerequisites

- Python 3.9 or higher
- Voice.AI API key (get one at [voice.ai](https://voice.ai))

### Install

Clone the repository and install in editable mode:

```bash
git clone https://github.com/voice-ai/pipecat-voice-ai.git
cd pipecat-voice-ai
pip install -e .
```

## Quick Start

```python
from pipecat_voice_ai import VoiceAiTTSService
from pipecat.transcriptions.language import Language

# Initialize the service
tts = VoiceAiTTSService(
    api_key="vk_your-api-key",
    voice_id="your-voice-id",  # Optional: uses default if not provided
    params=VoiceAiTTSService.InputParams(
        language=Language.EN,
        temperature=1.0,
        top_p=0.8
    )
)

# Use in a Pipecat pipeline
from pipecat.pipeline.pipeline import Pipeline

pipeline = Pipeline([
    # ... other processors ...
    tts,
    transport.output(),
])
```

## Configuration

### Supported Languages

Use Pipecat's `Language` enum:

```python
Language.EN  # English
Language.CA  # Catalan
Language.SV  # Swedish
Language.ES  # Spanish
Language.FR  # French
Language.DE  # German
Language.IT  # Italian
Language.PT  # Portuguese
Language.PL  # Polish
Language.RU  # Russian
Language.NL  # Dutch
```

### Parameters

```python
params = VoiceAiTTSService.InputParams(
    language=Language.EN,           # Target language
    model="voiceai-tts-v1-latest",  # TTS model (auto-selected if not specified)
    audio_format="pcm",             # Audio format (raw PCM)
    temperature=1.0,                # Creativity (0.0-2.0, default: 1.0)
    top_p=0.8,                      # Diversity (0.0-1.0, default: 0.8)
)
```

## Running the Examples

### Setup

1. Copy `.env.example` to `.env` and add your API key:

```bash
VOICEAI_API_KEY=vk_your-api-key-here
VOICEAI_VOICE_ID=your-voice-id-here  # Optional
```

2. Run the examples:

### Basic Example

Tests the service by generating audio to a file:

```bash
python examples/simple_tts.py
```

### Interactive Example

Full conversational bot with microphone input:

```bash
pip install pipecat-ai[local,openai,silero]  # Additional dependencies
python examples/microphone_example.py
```

Requires `OPENAI_API_KEY` in `.env` for speech-to-text and LLM services.

## Troubleshooting

**Authentication Errors**: Verify your API key starts with `vk_` and is properly set in `.env`

**No Audio Generated**: Try without specifying a `voice_id` to use the default voice

**Import Errors**: Ensure the package is installed with `pip install -e .`

For more help, open an issue or join the [Pipecat Discord](https://discord.gg/pipecat) `#community-integrations` channel.

## Technical Details

- **Base Class**: `InterruptibleTTSService` from Pipecat
- **Connection**: Persistent WebSocket with automatic reconnection
- **Audio Format**: Raw PCM at 32kHz mono (no decoding libraries needed)
- **Tested with**: Pipecat v0.0.101+

## Links

- [Voice.AI Website](https://voice.ai)
- [Voice.AI API Documentation](https://voice.ai/docs/api-reference/text-to-speech/single-context-websocket)
- [Pipecat Framework](https://github.com/pipecat-ai/pipecat)
- [Pipecat Documentation](https://docs.pipecat.ai)
- [Pipecat Discord](https://discord.gg/pipecat)

## Contributing

Contributions are welcome! Please ensure:
- Code follows Pipecat's conventions
- Changes are tested with the examples
- README is updated for new features

## Maintainer

This integration is officially maintained by [Voice.AI](https://voice.ai).

## License

BSD 2-Clause License (same as Pipecat)

See [CHANGELOG.md](CHANGELOG.md) for version history.
