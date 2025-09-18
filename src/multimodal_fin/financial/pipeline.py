import pandas as pd
from multimodal_fin.financial.event import Event


class EventPipeline:
    """Process multiple events in batch."""

    def __init__(self, conference_file: str, folder_path: str):
        self.df_conf = pd.read_csv(conference_file, index_col=0)
        self.folder_path = folder_path

    def run(self, ticker_filter: str = None) -> list[Event]:
        """Run pipeline for all events.

        Args:
            ticker_filter (str, optional): Filter events by ticker.

        Returns:
            list[Event]: List of processed Event objects.
        """
        df = self.df_conf
        if ticker_filter:
            df = df[df["symbol"] == ticker_filter]

        events = []
        for ts, symbol in zip(df["timestamp"], df["symbol"]):
            event = Event(event_date=ts, ticker=symbol, folder_path=self.folder_path)
            events.append(event)
        return events
