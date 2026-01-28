#
# Copyright (c) 2026, Voice.AI
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Voice.AI text-to-speech service implementation."""

import asyncio
import base64
import json
from typing import Any, AsyncGenerator, Mapping, Optional

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    AggregationType,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Voice.AI, you need to `pip install websockets`. "
        "See requirements.txt for details."
    )
    raise Exception(f"Missing module: {e}")


def language_to_voiceai_language(language: Language) -> Optional[str]:
    """Convert Pipecat Language enum to Voice.AI language codes.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Voice.AI language code (ISO 639-1 format), or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.CA: "ca",  # Catalan
        Language.DE: "de",  # German
        Language.EN: "en",  # English
        Language.ES: "es",  # Spanish
        Language.FR: "fr",  # French
        Language.IT: "it",  # Italian
        Language.NL: "nl",  # Dutch
        Language.PL: "pl",  # Polish
        Language.PT: "pt",  # Portuguese
        Language.RU: "ru",  # Russian
        Language.SV: "sv",  # Swedish
    }

    return LANGUAGE_MAP.get(language)


class VoiceAiTTSService(InterruptibleTTSService):
    """Text-to-speech service using Voice.AI's WebSocket API.

    Converts text to speech using Voice.AI's TTS models with support for multiple
    languages. Creates a new WebSocket connection for each TTS request.

    Supported features:

    - Multiple language support (en, ca, sv, es, fr, de, it, pt, pl, ru, nl)
    - Configurable voice selection
    - Temperature and top_p control for generation variety
    - Raw PCM audio output at 32kHz mono
    - Multiple TTS model options
    - Automatic interruption handling

    Example::

        tts = VoiceAiTTSService(
            api_key="vk_your-api-key",
            voice_id="your-voice-id",
            params=VoiceAiTTSService.InputParams(
                language=Language.EN,
                temperature=1.0,
                top_p=0.8
            )
        )
    """

    class InputParams(BaseModel):
        """Configuration parameters for Voice.AI TTS.

        Parameters:
            language: Target language for synthesis. Supported languages include English (en),
                Catalan (ca), Swedish (sv), Spanish (es), French (fr), German (de),
                Italian (it), Portuguese (pt), Polish (pl), Russian (ru), Dutch (nl).
                Defaults to English (en).
            model: TTS model to use. Supported models include voiceai-tts-v1-latest,
                voiceai-tts-v1-2025-12-19 (English only), voiceai-tts-multilingual-v1-latest,
                voiceai-tts-multilingual-v1-2025-01-14 (multilingual). If not provided,
                automatically selected based on language. English uses non-multilingual models;
                other languages use multilingual models.
            audio_format: Audio format for output. Supported formats include "pcm" for raw PCM
                audio. Defaults to "pcm".
            temperature: Temperature for generation (0.0-2.0). Higher values produce more
                random output. Defaults to 1.0.
            top_p: Top-p sampling parameter (0.0-1.0). Controls diversity of output.
                Defaults to 0.8.
        """

        language: Language = Language.EN
        model: Optional[str] = None
        audio_format: str = "pcm"
        temperature: float = Field(default=1.0, ge=0.0, le=2.0)
        top_p: float = Field(default=0.8, ge=0.0, le=1.0)

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: Optional[str] = None,
        url: str = "wss://dev.voice.ai/api/v1/tts/stream",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Voice.AI TTS service.

        Args:
            api_key: Voice.AI API key for authentication (format: vk_*).
            voice_id: Voice identifier for synthesis. If not provided, uses default built-in voice.
            url: WebSocket URL for Voice.AI TTS API (default production URL).
            sample_rate: Output audio sample rate. Defaults to 32000 Hz (Voice.AI native rate).
            params: Optional input parameters to configure voice synthesis settings.
            **kwargs: Additional keyword arguments passed to InterruptibleTTSService base class.
        """
        super().__init__(
            push_text_frames=True,
            pause_frame_processing=True,
            push_stop_frames=False,  # We push TTSStoppedFrame manually on completion
            sample_rate=sample_rate or 32000,
            **kwargs,
        )

        self._api_key = api_key
        self._voice_id = voice_id
        self._url = url
        self._settings = {}

        # Register voice for service tracking
        if voice_id:
            self.set_voice(voice_id)

        # WebSocket state
        self._websocket = None
        self._receive_task = None
        self._started = False
        self._disconnecting = False
        
        # Synchronization for audio completion
        self._audio_completion_event = None

        # Set up parameters
        if params:
            self._settings["language"] = language_to_voiceai_language(params.language) or "en"
            self._settings["model"] = params.model
            self._settings["audio_format"] = params.audio_format
            self._settings["temperature"] = params.temperature
            self._settings["top_p"] = params.top_p
        else:
            # Defaults
            self._settings["language"] = "en"
            self._settings["model"] = None
            self._settings["audio_format"] = "pcm"
            self._settings["temperature"] = 1.0
            self._settings["top_p"] = 0.8

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Voice.AI service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Voice.AI language format.

        Args:
            language: The language to convert.

        Returns:
            The Voice.AI-specific language code (ISO 639-1), or None if not supported.
        """
        return language_to_voiceai_language(language)

    async def start(self, frame: StartFrame):
        """Start the Voice.AI TTS service and establish WebSocket connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Voice.AI TTS service and close WebSocket connection.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Voice.AI TTS service and close WebSocket connection.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Update service settings and reconnect with new configuration.

        Args:
            settings: Dictionary of settings to update.
        """
        await super()._update_settings(settings)
        # Reconnect to apply new settings
        logger.info(f"Reconnecting Voice.AI TTS with updated settings")
        await self._disconnect()
        await self._connect()

    async def _connect(self):
        """Connect to Voice.AI WebSocket and start background tasks."""
        await super()._connect()
        
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from Voice.AI WebSocket and clean up tasks."""
        await super()._disconnect()

        try:
            self._disconnecting = True

            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=2.0)
                self._receive_task = None

            await self._disconnect_websocket()

        except Exception as e:
            await self.push_error(error_msg=f"Error during disconnect: {e}", exception=e)
        finally:
            self._started = False
            self._websocket = None
            self._disconnecting = False

    async def _connect_websocket(self):
        """Establish WebSocket connection and send initialization message."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            # Connect with authentication
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websocket_connect(self._url, additional_headers=headers)
            
            logger.debug("Connected to Voice.AI WebSocket")

            # Send initialization message (settings only, no text)
            init_message = {
                "audio_format": self._settings["audio_format"],
                "temperature": self._settings["temperature"],
                "top_p": self._settings["top_p"],
                "language": self._settings["language"],
            }

            # Add optional fields
            if self._voice_id:
                init_message["voice_id"] = self._voice_id
            if self._settings["model"]:
                init_message["model"] = self._settings["model"]

            await self._websocket.send(json.dumps(init_message))
            logger.debug("Sent initialization message to Voice.AI")

            await self._call_event_handler("on_connected")

        except Exception as e:
            await self.push_error(
                error_msg=f"Error connecting to Voice.AI WebSocket: {e}", exception=e
            )
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Voice.AI")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._started = False
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _cleanup_connection(self):
        """Clean up closed connection after request completion."""
        try:
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None
            
            if self._websocket:
                try:
                    await self._websocket.close()
                except:
                    pass
                self._websocket = None
                
            self._started = False
        except Exception as e:
            logger.debug(f"Error during connection cleanup: {e}")

    def _get_websocket(self):
        """Get the current websocket connection.

        Returns:
            The active websocket connection.

        Raises:
            Exception: If websocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("WebSocket not connected")

    async def _receive_messages(self):
        """Receive and process messages from Voice.AI WebSocket."""
        async for message in self._get_websocket():
            if isinstance(message, str):
                msg = json.loads(message)

                # Handle audio chunk
                if "audio" in msg:
                    await self.stop_ttfb_metrics()
                    
                    # Decode base64 to get raw PCM audio bytes
                    audio_data = base64.b64decode(msg["audio"])
                    
                    # Voice.AI returns PCM audio (16-bit samples, mono)
                    frame = TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )
                    await self.push_frame(frame)

                # Handle completion signal
                elif msg.get("is_last"):
                    logger.debug("Received completion signal from Voice.AI")
                    # Signal that this text chunk is complete
                    await self.push_frame(TTSStoppedFrame())
                    self._started = False
                    
                    # Signal completion to run_tts() if waiting
                    if self._audio_completion_event:
                        self._audio_completion_event.set()

                # Handle error
                elif "error" in msg:
                    error_msg = msg["error"]
                    await self.push_error(error_msg=f"TTS Error: {error_msg}")
                    await self.push_frame(ErrorFrame(error=f"Voice.AI TTS error: {error_msg}"))
                    
                    # Signal completion on error too
                    if self._audio_completion_event:
                        self._audio_completion_event.set()

    async def _receive_task_handler(self, report_error):
        """Background task to receive messages from WebSocket.

        Args:
            report_error: Callback to report errors.
        """
        try:
            await self._receive_messages()
        except Exception as e:
            if not self._disconnecting:
                logger.error(f"Error in receive task: {e}")
                await report_error(ErrorFrame(error=f"Voice.AI receive error: {e}", exception=e))

    async def _send_text(self, text: str):
        """Send text-only message to Voice.AI for synthesis.

        Args:
            text: The text to synthesize.
        """
        if self._disconnecting:
            logger.warning("Service is disconnecting, ignoring text send")
            return

        if self._websocket and self._websocket.state == State.OPEN:
            msg = {
                "text": text,
                "flush": True,  # Trigger audio generation
            }
            await self._websocket.send(json.dumps(msg))
            logger.debug(f"Sent text to Voice.AI: {text[:50]}...")
        else:
            logger.warning("WebSocket not ready, cannot send text")
            raise Exception("WebSocket not connected")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Voice.AI's WebSocket API.

        Creates a WebSocket connection, sends text for synthesis, and receives audio.
        Connection closes automatically after completion.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: TTSStartedFrame and TTSTextFrame. Audio frames are pushed via
                the receive task, and TTSStoppedFrame is sent after completion.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            # Ensure we're connected
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            await self.start_ttfb_metrics()

            # Mark that we started speaking
            if not self._started:
                self._started = True

            # Create completion event for this request
            self._audio_completion_event = asyncio.Event()

            # Yield start frames
            yield TTSStartedFrame()
            yield TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)

            # Send text-only message (settings already sent in init)
            await self._send_text(text)

            # Wait for audio to complete (is_last signal received)
            await self._audio_completion_event.wait()
            
            logger.debug(f"Audio generation complete for: {text[:50]}...")

            await self.start_tts_usage_metrics(text)
            
            # Voice.AI closes connection after each request
            await self._cleanup_connection()

        except Exception as e:
            logger.error(f"Error in Voice.AI TTS: {e}")
            yield ErrorFrame(error=f"Voice.AI TTS error: {e}", exception=e)
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
            await self._cleanup_connection()
        finally:
            self._audio_completion_event = None

    async def _report_error(self, error: ErrorFrame):
        """Report errors from background tasks.

        Args:
            error: The error frame to report.
        """
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error_frame(error)

