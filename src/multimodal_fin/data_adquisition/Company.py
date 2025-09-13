"""Company-level data acquisition logic for transcripts and audio.

Uses the `earningscall` library to retrieve and store financial conference data.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from earningscall import get_company
from earningscall.company import Company as EarningsCompany

from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


class CompanyDataAcquisition:
    """Fetches and stores all available transcripts and audio for a given company."""

    def __init__(self, company_code: str):
        """
        Initialize the acquisition interface for a specific company.

        Args:
            company_code (str): Ticker symbol or code for the company.
        """
        self.company_code = company_code.lower()
        self.company: EarningsCompany = self._initialize_company()

    def _initialize_company(self) -> EarningsCompany:
        """
        Retrieve the company object from the earningscall API.

        Returns:
            EarningsCompany: Initialized company object.
        """
        return get_company(self.company_code)

    def get_and_save_one_transcript(
        self, base_path: str, year: int, quarter: int, level: int = 3
    ) -> None:
        """
        Fetch and save a single transcript for a given year and quarter.

        Args:
            base_path (str): Base directory to save the transcript.
            year (int): Year of the earnings call.
            quarter (int): Quarter of the earnings call.
            level (int): Transcript level to fetch (default: 3).
        """
        logger.info(f"Fetching transcript for {self.company_code.upper()} Q{quarter} {year}")
        transcript = self.company.get_transcript(year=year, quarter=quarter, level=level)

        output_path = Path(base_path) / self.company_code.upper() / str(year) / f"Q{quarter}"

        if transcript:
            self.save_transcripts_json({f"LEVEL_{level}": transcript}, output_path)
            logger.info(f"Transcript found and saved. Q{quarter} {year}. [OK]")
        else:
            logger.warning(f"No transcript found. Q{quarter} {year}. [NOT FOUND]")

    def get_and_save_all_transcripts_and_audio(self, base_path: str, level: int = 4) -> None:
        """
        Fetch and store all available transcripts and audio files for the company.

        Args:
            base_path (str): Directory where results will be saved.
            level (int): Transcript level to prioritize (default: 4).
        """
        logger.info(f"Fetching all transcripts for {self.company_code.upper()}...")

        for event in self.company.events():
            if datetime.now().timestamp() < event.conference_date.timestamp():
                logger.info(
                    f"* Skipping future event for {self.company_code.upper()} Q{event.quarter} {event.year}"
                )
                continue

            transcripts = {}

            try:
                try:
                    transcripts['LEVEL_4'] = self.company.get_transcript(event=event, level=4)
                except Exception as e:
                    logger.warning(f"Failed to retrieve LEVEL_4: {e}", exc_info=True)

                try:
                    transcripts['LEVEL_3'] = self.company.get_transcript(event=event, level=3)
                except Exception as e:
                    logger.warning(f"Failed to retrieve LEVEL_3: {e}", exc_info=True)

                output_path = Path(base_path) / self.company_code.upper() / str(event.year) / f"Q{event.quarter}"

                if transcripts:
                    self.save_transcripts(transcripts, output_path)
                    try:
                        self.company.download_audio_file(event=event, file_name=str(output_path / 'audio.mp3'))
                    except Exception as e:
                        logger.warning(f"Audio download failed for Q{event.quarter} {event.year}: {e}", exc_info=True)
                    logger.info(f"Transcript and audio saved. Q{event.quarter} {event.year}. [OK]")
                else:
                    logger.warning(f"No transcript found. Q{event.quarter} {event.year}. [NOT FOUND]")

            except Exception as e:
                logger.error(f"Unexpected error processing Q{event.quarter} {event.year}: {e}", exc_info=True)

        logger.info("-" * 80)

    def save_transcripts(self, transcripts: dict, path: Path) -> None:
        """
        Save transcripts in both JSON and CSV formats.

        Args:
            transcripts (dict): Dictionary with transcript objects by level.
            path (Path): Output directory path.
        """
        self.save_transcripts_json(transcripts, path)
        self.save_transcript_csv(transcripts, path)

    def save_transcripts_json(self, transcripts: dict, path: Path) -> None:
        """
        Serialize and save transcripts as JSON files.

        Args:
            transcripts (dict): Dictionary of transcript objects by level.
            path (Path): Output directory.
        """
        path.mkdir(parents=True, exist_ok=True)

        for level, transcript in transcripts.items():
            file_path = path / f"{level}.json"
            try:
                with file_path.open("w", encoding="utf-8") as f:
                    json.dump(transcript.to_dict(), f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to save {level} transcript: {e}", exc_info=True)

    def save_transcript_csv(self, transcripts: dict, path: Path) -> None:
        """
        Convert and save LEVEL_3 transcript as a CSV file.

        Args:
            transcripts (dict): Transcript dictionary (must include 'LEVEL_3').
            path (Path): Output directory.
        """
        if 'LEVEL_3' not in transcripts:
            logger.warning("No LEVEL_3 transcript available to save as CSV.")
            return

        try:
            speakers = transcripts['LEVEL_3'].to_dict().get("speakers", [])
            rows = []

            for speaker in speakers:
                speaker_info = speaker.get("speaker_info", {})
                row = {
                    "speaker_id": speaker.get("speaker"),
                    "name": speaker_info.get("name"),
                    "title": speaker_info.get("title"),
                    "text": speaker.get("text"),
                    "start_time": speaker.get("start_times", [None])[0],
                    "end_time": speaker.get("start_times", [None])[-1],
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(path / "transcript.csv", index=False, encoding="utf-8")

        except Exception as e:
            logger.error("Failed to save transcript CSV", exc_info=True)
