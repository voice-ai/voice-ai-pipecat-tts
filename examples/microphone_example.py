#!/usr/bin/env python3
"""
Voice.ai TTS Service with Live Microphone Input

This example demonstrates the Voice.ai TTS service in a full conversational pipeline
using your local microphone and speakers. Talk to the bot and hear it respond
using Voice.ai TTS!

Requirements:
    - pip install pipecat-ai[local,openai,silero]
    - On Windows: PyAudio should install automatically
    - On macOS: brew install portaudio
    - On Linux: sudo apt-get install portaudio19-dev python3-pyaudio

Setup:
    1. Copy .env.example to .env
    2. Add your API keys:
       - VOICEAI_API_KEY=your_key_here
       - OPENAI_API_KEY=your_key_here
    3. Run: python examples/microphone_example.py
    4. Speak into your microphone!
    5. Press Ctrl+C to exit
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# Pipecat core
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

# Services
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService

# Local audio transport
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

# Voice.ai TTS service
from pipecat_voice_ai.tts import VoiceAiTTSService

load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    """Run the conversational bot with microphone input."""
    logger.info("=" * 70)
    logger.info("Voice.ai TTS Microphone Test")
    logger.info("=" * 70)

    # Validate API keys
    voiceai_api_key = os.getenv("VOICEAI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not voiceai_api_key:
        logger.error("[ERROR] VOICEAI_API_KEY not found in environment")
        logger.error("   Copy .env.example to .env and add your key")
        return

    if not openai_api_key:
        logger.error("[ERROR] OPENAI_API_KEY not found in environment")
        logger.error("   Add to .env: OPENAI_API_KEY=...")
        return

    logger.info("[OK] API keys found")

    # Configure Local Audio Transport (uses system microphone/speakers)
    logger.info("Setting up local audio transport...")
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,  # Enable microphone input
            audio_out_enabled=True,  # Enable speaker output
            audio_out_sample_rate=32000,  # Match Voice.ai's 32kHz output
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.5  # Wait 0.5s of silence before considering speech ended
                )
            ),
        )
    )
    logger.info("[OK] Local audio transport configured")

    # Initialize services
    logger.info("Initializing services...")

    # 1. Speech-to-Text: OpenAI Whisper
    stt = OpenAISTTService(api_key=openai_api_key, model="whisper-1")
    logger.info("[OK] STT: OpenAI Whisper")

    # 2. Language Model: OpenAI GPT
    llm = OpenAILLMService(api_key=openai_api_key, model="gpt-4o-mini")
    logger.info("[OK] LLM: OpenAI GPT-4o-mini")

    # 3. Text-to-Speech: Voice.ai
    tts = VoiceAiTTSService(
        api_key=voiceai_api_key,
        voice_id=os.getenv("VOICEAI_VOICE_ID"),  # Optional custom voice
        params=VoiceAiTTSService.InputParams(
            temperature=1.0,
            top_p=0.8,
        ),
    )
    logger.info("[OK] TTS: Voice.ai")

    # Define the conversation context
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant. Your responses will be spoken aloud using "
                "Voice.ai text-to-speech, so keep your answers concise and conversational. "
                "Avoid using special characters, bullet points, or formatting that doesn't "
                "work well in speech. Be friendly and engaging."
            ),
        }
    ]

    # Create LLM context and aggregators
    # Using simple aggregators without advanced turn analysis
    # (LocalSmartTurnAnalyzerV3 requires additional dependencies)
    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    # Build the pipeline
    logger.info("Building pipeline...")
    pipeline = Pipeline(
        [
            transport.input(),  # Microphone input
            stt,  # Speech-to-text
            user_aggregator,  # User turn aggregation
            llm,  # Language model
            tts,  # Text-to-speech (Voice.ai)
            transport.output(),  # Speaker output
            assistant_aggregator,  # Assistant turn aggregation
        ]
    )
    logger.info("[OK] Pipeline built")

    # Create the task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Start the conversation with a greeting
    messages.append(
        {
            "role": "system",
            "content": "Please introduce yourself briefly to the user and ask how you can help.",
        }
    )
    await task.queue_frames([LLMRunFrame()])

    # Run the bot
    logger.info("=" * 70)
    logger.info("Bot is ready! Start speaking into your microphone...")
    logger.info("The bot will respond using Voice.ai TTS")
    logger.info("Press Ctrl+C to exit")
    logger.info("=" * 70)

    runner = PipelineRunner()

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    finally:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())

