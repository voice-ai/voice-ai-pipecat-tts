#!/usr/bin/env python
#
# Simple test script for Voice.ai TTS service
#
# This script tests the Voice.ai TTS service without requiring a full transport setup.
# It generates raw PCM audio and saves it to a file for verification.
#

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, ErrorFrame
from pipecat_voice_ai.tts import VoiceAiTTSService

load_dotenv()


async def test_voiceai():
    """Simple test of Voice.ai TTS service."""

    # Check for API key
    api_key = os.getenv("VOICEAI_API_KEY")
    if not api_key:
        logger.error("VOICEAI_API_KEY not found in environment!")
        logger.error("Please create a .env file with your API key:")
        logger.error("  VOICEAI_API_KEY=vk_your_api_key_here")
        return

    logger.info("Initializing Voice.ai TTS service...")

    # Initialize the service
    tts = VoiceAiTTSService(
        api_key=api_key,
        voice_id=os.getenv("VOICEAI_VOICE_ID"),  # Optional
        params=VoiceAiTTSService.InputParams(
            temperature=1.0,
            top_p=0.8,
        ),
    )

    # Test text
    text = "Hello! This is a test of the Voice.ai TTS API."
    logger.info(f"Generating speech for: '{text}'")

    # Generate speech
    audio_chunks = []
    started = False
    stopped = False
    error_occurred = False

    try:
        async for frame in tts.run_tts(text):
            if isinstance(frame, TTSStartedFrame):
                started = True
                logger.info("[OK] TTS started")

            elif isinstance(frame, TTSAudioRawFrame):
                logger.info(
                    f"[OK] Received audio chunk: {len(frame.audio)} bytes "
                    f"(sample_rate={frame.sample_rate}, channels={frame.num_channels})"
                )
                audio_chunks.append(frame.audio)

            elif isinstance(frame, TTSStoppedFrame):
                stopped = True
                logger.info("[OK] TTS stopped")

            elif isinstance(frame, ErrorFrame):
                error_occurred = True
                logger.error(f"[ERROR] {frame.error}")

    except Exception as e:
        logger.error(f"[ERROR] Exception during TTS: {e}")
        import traceback

        traceback.print_exc()
        return

    # Report results
    logger.info("\n" + "=" * 60)
    logger.info("Test Results:")
    logger.info("=" * 60)
    logger.info(f"Started: {started}")
    logger.info(f"Stopped: {stopped}")
    logger.info(f"Audio chunks received: {len(audio_chunks)}")
    logger.info(f"Error occurred: {error_occurred}")

    # Save to file
    if audio_chunks:
        output_file = "voiceai_test_output.pcm"
        total_bytes = sum(len(chunk) for chunk in audio_chunks)

        with open(output_file, "wb") as f:
            for chunk in audio_chunks:
                f.write(chunk)

        logger.info(f"\n[SUCCESS] Saved {total_bytes} bytes to {output_file}")
        logger.info(f"  Raw PCM audio (32kHz, 16-bit, mono)")
        logger.info(f"  To play: ffplay -f s16le -ar 32000 -ac 1 {output_file}")
    else:
        logger.warning("\n[FAILURE] No audio data was received!")
        logger.warning("  Check your API key and network connection.")

    logger.info("=" * 60)


if __name__ == "__main__":
    logger.info("Voice.ai TTS Service - Simple Test")
    logger.info("=" * 60)
    asyncio.run(test_voiceai())

