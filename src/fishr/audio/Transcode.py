from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Literal

AudioFormat = Literal[
    "ogg",
    "wav",
    "mp3",
    "flac",
    "opus",
    "aac",
    "m4a",
]

CodecForFormat = {
    "ogg": "libvorbis",
    "wav": "pcm_s16le",
    "mp3": "libmp3lame",
    "flac": "flac",
    "opus": "libopus",
    "aac": "aac",
    "m4a": "aac",
}

ExtForFormat = {
    "ogg": "ogg",
    "wav": "wav",
    "mp3": "mp3",
    "flac": "flac",
    "opus": "opus",
    "aac": "aac",
    "m4a": "m4a",
}

MimeForFormat = {
    "ogg": "audio/ogg",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "m4a": "audio/mp4",
}

Log = logging.getLogger("fishr.audio.transcode")


def Transcode(
    Audio: bytes,
    Format: AudioFormat,
    *,
    SourceMime: str = "audio/mpeg",
) -> tuple[bytes, str]:
    """Transcode raw audio bytes into ``Format`` using the bundled static ffmpeg.

    Returns ``(data, mime_type)``.

    ``static-ffmpeg`` ships its own ffmpeg binary, so no system ffmpeg is
    needed. It is imported lazily so startup is not penalized when
    transcoding is unused.
    """
    if Format not in CodecForFormat:
        raise ValueError(f"unsupported format: {Format}")

    if not Audio:
        Log.warning("transcode skipped: empty input audio")
        return b"", MimeForFormat[Format]

    SourceExt = _ExtFromMime(SourceMime)
    SrcDir = tempfile.mkdtemp(prefix="fishr_tts_src_")
    SrcPath = os.path.join(SrcDir, f"in.{SourceExt}")
    OutPath = os.path.join(SrcDir, f"out.{ExtForFormat[Format]}")
    with open(SrcPath, "wb") as Fh:
        Fh.write(Audio)

    try:
        from static_ffmpeg import run

        Ffmpeg, _ = run.get_or_fetch_platform_executables_else_raise()
        try:
            subprocess.run(
                [
                    Ffmpeg,
                    "-y",
                    "-i",
                    SrcPath,
                    "-c:a",
                    CodecForFormat[Format],
                    OutPath,
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as Exc:
            Stderr = (Exc.stderr or b"").decode(errors="replace")[:200]
            Log.warning(
                "ffmpeg transcode failed (exit %s): %s",
                Exc.returncode,
                Stderr,
            )
            return b"", MimeForFormat[Format]
        if not os.path.exists(OutPath):
            Log.warning("ffmpeg produced no output for format=%s", Format)
            return b"", MimeForFormat[Format]
        with open(OutPath, "rb") as Fh:
            return Fh.read(), MimeForFormat[Format]
    finally:
        for Path in (SrcPath, OutPath):
            try:
                os.remove(Path)
            except OSError:
                pass
        try:
            os.rmdir(SrcDir)
        except OSError:
            pass


def _ExtFromMime(Mime: str) -> str:
    Mime = Mime.split(";", 1)[0].strip().lower()
    if Mime in ("audio/mpeg", "audio/mp3"):
        return "mp3"
    if Mime in ("audio/ogg",):
        return "ogg"
    if Mime in ("audio/wav", "audio/x-wav", "audio/wave"):
        return "wav"
    if Mime in ("audio/flac",):
        return "flac"
    if Mime in ("audio/opus",):
        return "opus"
    if Mime in ("audio/aac",):
        return "aac"
    if Mime in ("audio/mp4", "audio/m4a"):
        return "m4a"
    return "mp3"


__all__ = [
    "Transcode",
    "AudioFormat",
]
