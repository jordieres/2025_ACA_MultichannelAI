from datetime import datetime
import earningscall
from earningscall import get_company
from pathlib import Path
import json
import os

class Company:
    def __init__(self, company_code: str):
        self.company_code = company_code.lower()
        self.company = self._initialize_company()

    def _initialize_company(self) -> earningscall.company.Company:
        """
        Initializes the company object using the given company code.
        """
        return get_company(self.company_code)

    def get_and_save_one_transcript(self, base_path: str, year: int, quarter: int, level=3) -> None:
        """
        Fetches and saves a single earnings call transcript for the specified year and quarter.
        """
        print(f"Fetching transcript for {self.company_code.upper()} Q{quarter} {year}")
        transcript = self.company.get_transcript(year=year, quarter=quarter, level=level)

        path = f'{base_path}/{self.company_code.upper()}/{year}/{quarter}'

        if transcript:
            self.save_transcript(transcript, path)
            print(f"Transcript found and loaded. Q{quarter} {year}. [OK]")
        else:
            print(f"No transcript found. Q{quarter} {year}. [ERROR]")

    def get_and_save_all_transcripts_and_audio(self, base_path: str, level=3) -> None:
        """
        Fetches and saves all available earnings call transcripts for the company.
        """
        print(f"Fetching all transcripts for {self.company_code.upper()}..")

        for event in self.company.events():
            # Skip future events
            if datetime.now().timestamp() < event.conference_date.timestamp():
                print(f"* {self.company.company_info.symbol} Q{event.quarter} {event.year} -- skipping, conference date in the future")
                continue

            try:
                transcript = self.company.get_transcript(event=event, level=level)

                path = f'{base_path}/{self.company_code.upper()}/{event.year}/Q{event.quarter}'

                if transcript:
                    self.save_transcript(transcript, path)
                    self.company.download_audio_file(event=event, file_name=path + '/audio.mp3')
                    print(f"Transcript and audio found and loaded. Q{event.quarter} {event.year} [OK]")
                else:
                    print(f"No transcript found. Q{event.quarter} {event.year}. [ERROR]")
            except Exception as e:
                print(f"Error processing Q{event.quarter} {event.year} for {self.company_code.upper()}: {e}")
        print("-" * 150)
    
    @staticmethod
    def save_transcript(transcript, path: str) -> None:
        """
        Saves the transcript dictionary to a JSON file.
        """
        # Ensure the directory exists
        Path(path).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(path, 'text.json')
        
        try:
            with open(file_path, "w", encoding="utf-8") as archivo:
                json.dump(transcript.to_dict(), archivo, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save transcript: {e}")
