"""
Data Preprocessing Module for ElectriQ
Handles audio preprocessing, ASR transcription, and speaker diarization.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Audio processing
try:
    import librosa
    import soundfile as sf

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logging.warning("Audio processing libraries not available")

# ASR
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("Whisper ASR not available")

# Speaker diarization
try:
    from pyannote.audio import Pipeline

    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    logging.warning("Speaker diarization not available")


@dataclass
class AudioSegment:
    """Represents a segmented audio clip with metadata."""
    audio_path: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    transcript: Optional[str] = None
    sample_rate: int = 16000


@dataclass
class ProcessedDialogue:
    """Represents a processed dialogue with all metadata."""
    dialogue_id: str
    segments: List[AudioSegment]
    full_transcript: str
    metadata: Dict
    quality_score: float = 0.0


class AudioPreprocessor:
    """
    Audio preprocessing for dialogue data.
    Handles loading, resampling, and quality filtering.
    """

    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize audio preprocessor.

        Args:
            target_sample_rate: Target sampling rate for all audio
        """
        self.target_sample_rate = target_sample_rate

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample if necessary.

        Args:
            audio_path: Path to audio file

        Returns:
            tuple: (audio_array, sample_rate)
        """
        if not AUDIO_AVAILABLE:
            raise ImportError("librosa not installed")

        audio, sr = librosa.load(audio_path, sr=None)

        # Resample if necessary
        if sr != self.target_sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.target_sample_rate
            )

        return audio, self.target_sample_rate

    def split_audio(
            self,
            audio_path: str,
            segment_duration: float = 30.0,
            overlap: float = 0.0
    ) -> List[AudioSegment]:
        """
        Split long audio into segments.

        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments in seconds

        Returns:
            list: List of AudioSegment objects
        """
        audio, sr = self.load_audio(audio_path)
        duration = len(audio) / sr

        segments = []
        step = segment_duration - overlap
        start = 0.0

        while start < duration:
            end = min(start + segment_duration, duration)

            # Extract segment
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]

            # Save segment
            segment_path = f"{audio_path}.seg_{len(segments)}.wav"
            sf.write(segment_path, segment_audio, sr)

            segments.append(AudioSegment(
                audio_path=segment_path,
                start_time=start,
                end_time=end,
                sample_rate=sr
            ))

            start += step
            if end >= duration:
                break

        return segments

    def filter_by_quality(
            self,
            audio_path: str,
            min_duration: float = 5.0,
            max_duration: float = 300.0,
            min_snr: float = 10.0
    ) -> bool:
        """
        Filter audio by quality metrics.

        Args:
            audio_path: Path to audio file
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            min_snr: Minimum signal-to-noise ratio in dB

        Returns:
            bool: True if audio passes quality filter
        """
        if not AUDIO_AVAILABLE:
            return True

        audio, sr = self.load_audio(audio_path)
        duration = len(audio) / sr

        # Check duration
        if duration < min_duration or duration > max_duration:
            return False

        # Estimate SNR (simplified)
        signal_power = np.mean(audio ** 2)
        noise_estimate = np.percentile(np.abs(audio), 10)
        noise_power = noise_estimate ** 2

        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            if snr < min_snr:
                return False

        return True


class ASRTranscriber:
    """
    Automatic Speech Recognition using Whisper model.
    """

    def __init__(self, model_name: str = "openai/whisper-base"):
        """
        Initialize ASR transcriber.

        Args:
            model_name: Whisper model name
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper transformers not installed")

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model_name = model_name

    def transcribe(self, audio_path: str, language: str = "zh") -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code for transcription

        Returns:
            str: Transcribed text
        """
        import torch

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Process audio
        inputs = self.processor(
            audio,
            return_tensors="pt",
            sampling_rate=16000
        )
        input_features = inputs.input_features

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        # Decode transcription
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription

    def transcribe_batch(
            self,
            audio_paths: List[str],
            language: str = "zh"
    ) -> List[str]:
        """
        Transcribe multiple audio files.

        Args:
            audio_paths: List of audio file paths
            language: Language code

        Returns:
            list: List of transcriptions
        """
        transcriptions = []
        for path in audio_paths:
            try:
                text = self.transcribe(path, language)
                transcriptions.append(text)
            except Exception as e:
                logging.error(f"Failed to transcribe {path}: {e}")
                transcriptions.append("")

        return transcriptions


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio.
    """

    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1"):
        """
        Initialize speaker diarizer.

        Args:
            model_name: Pyannote diarization model name
        """
        if not DIARIZATION_AVAILABLE:
            raise ImportError("pyannote.audio not installed")

        self.pipeline = Pipeline.from_pretrained(model_name)

    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            list: List of speaker segments with timestamps
        """
        diarization = self.pipeline(audio_path)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })

        return segments

    def assign_speakers_to_segments(
            self,
            segments: List[AudioSegment],
            diarization_result: List[Dict]
    ) -> List[AudioSegment]:
        """
        Assign speaker labels to audio segments.

        Args:
            segments: List of AudioSegment objects
            diarization_result: Diarization output

        Returns:
            list: Updated segments with speaker labels
        """
        for segment in segments:
            # Find overlapping diarization segments
            matching_speakers = []
            for diar_seg in diarization_result:
                if (segment.start_time <= diar_seg['end'] and
                        segment.end_time >= diar_seg['start']):
                    matching_speakers.append(diar_seg['speaker'])

            # Assign most common speaker
            if matching_speakers:
                segment.speaker = max(
                    set(matching_speakers),
                    key=matching_speakers.count
                )

        return segments


class DialoguePreprocessor:
    """
    Main dialogue preprocessing pipeline.
    Combines audio preprocessing, ASR, and diarization.
    """

    def __init__(
            self,
            audio_preprocessor: Optional[AudioPreprocessor] = None,
            asr_transcriber: Optional[ASRTranscriber] = None,
            speaker_diarizer: Optional[SpeakerDiarizer] = None
    ):
        """
        Initialize dialogue preprocessor.

        Args:
            audio_preprocessor: Audio preprocessing module
            asr_transcriber: ASR transcription module
            speaker_diarizer: Speaker diarization module
        """
        self.audio_preprocessor = audio_preprocessor or AudioPreprocessor()
        self.asr_transcriber = asr_transcriber or ASRTranscriber()
        self.speaker_diarizer = speaker_diarizer or SpeakerDiarizer()

    def process_dialogue(
            self,
            audio_path: str,
            dialogue_id: str,
            metadata: Optional[Dict] = None
    ) -> ProcessedDialogue:
        """
        Process a complete dialogue audio file.

        Args:
            audio_path: Path to dialogue audio
            dialogue_id: Unique identifier for dialogue
            metadata: Additional metadata

        Returns:
            ProcessedDialogue: Complete processed dialogue
        """
        # Check audio quality
        if not self.audio_preprocessor.filter_by_quality(audio_path):
            raise ValueError(f"Audio {audio_path} failed quality filter")

        # Split audio into segments
        segments = self.audio_preprocessor.split_audio(audio_path)

        # Perform speaker diarization
        diarization_result = self.speaker_diarizer.diarize(audio_path)
        segments = self.speaker_diarizer.assign_speakers_to_segments(
            segments,
            diarization_result
        )

        # Transcribe each segment
        for segment in segments:
            segment.transcript = self.asr_transcriber.transcribe(
                segment.audio_path
            )

        # Combine transcripts
        full_transcript = " ".join([
            f"[{seg.speaker or 'UNK'}]: {seg.transcript}"
            for seg in segments
        ])

        return ProcessedDialogue(
            dialogue_id=dialogue_id,
            segments=segments,
            full_transcript=full_transcript,
            metadata=metadata or {},
            quality_score=self._compute_quality_score(segments)
        )

    def _compute_quality_score(self, segments: List[AudioSegment]) -> float:
        """
        Compute overall quality score for dialogue.

        Args:
            segments: List of audio segments

        Returns:
            float: Quality score (0-1)
        """
        if not segments:
            return 0.0

        # Factors: transcription completeness, speaker coverage, segment quality
        transcript_lengths = [len(seg.transcript or "") for seg in segments]
        avg_length = np.mean(transcript_lengths)

        # Normalize to 0-1
        score = min(avg_length / 100.0, 1.0)

        return score

    def process_batch(
            self,
            audio_paths: List[str],
            dialogue_ids: List[str],
            metadata_list: Optional[List[Dict]] = None,
            output_dir: str = "./processed_dialogues"
    ) -> List[ProcessedDialogue]:
        """
        Process multiple dialogue files.

        Args:
            audio_paths: List of audio file paths
            dialogue_ids: List of dialogue IDs
            metadata_list: List of metadata dictionaries
            output_dir: Directory to save processed dialogues

        Returns:
            list: List of ProcessedDialogue objects
        """
        if metadata_list is None:
            metadata_list = [{} for _ in audio_paths]

        os.makedirs(output_dir, exist_ok=True)
        processed_dialogues = []

        for audio_path, dialogue_id, metadata in zip(
                audio_paths, dialogue_ids, metadata_list
        ):
            try:
                dialogue = self.process_dialogue(
                    audio_path, dialogue_id, metadata
                )
                processed_dialogues.append(dialogue)

                # Save to file
                output_path = os.path.join(output_dir, f"{dialogue_id}.json")
                self._save_dialogue(dialogue, output_path)

            except Exception as e:
                logging.error(f"Failed to process {audio_path}: {e}")

        return processed_dialogues

    def _save_dialogue(
            self,
            dialogue: ProcessedDialogue,
            output_path: str
    ):
        """
        Save processed dialogue to JSON file.

        Args:
            dialogue: ProcessedDialogue object
            output_path: Output file path
        """
        data = {
            'dialogue_id': dialogue.dialogue_id,
            'segments': [
                {
                    'audio_path': seg.audio_path,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'speaker': seg.speaker,
                    'transcript': seg.transcript,
                    'sample_rate': seg.sample_rate
                }
                for seg in dialogue.segments
            ],
            'full_transcript': dialogue.full_transcript,
            'metadata': dialogue.metadata,
            'quality_score': dialogue.quality_score
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # Example usage
    preprocessor = DialoguePreprocessor()

    # Process single dialogue
    dialogue = preprocessor.process_dialogue(
        audio_path="./data/raw/dialogue_001.wav",
        dialogue_id="dialogue_001",
        metadata={'source': 'call_center', 'date': '2024-01-15'}
    )

    print(f"Dialogue ID: {dialogue.dialogue_id}")
    print(f"Quality Score: {dialogue.quality_score:.2f}")
    print(f"Transcript: {dialogue.full_transcript[:200]}...")