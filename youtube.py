from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from youtube_transcript_api import YouTubeTranscriptApi


@dataclass
class Metadata(DataClassJsonMixin):
    start: float
    link: str


@dataclass
class Text:
    text: str
    start: float

    def to_metadata(self, video_uid: str) -> Metadata:
        link = f"https://www.youtube.com/watch?v={video_uid}&t={(self.start)}s"
        return Metadata(self.start, link)


@dataclass
class Transcript:
    texts: list[Text]


@dataclass
class YouTubeVideo:
    uid: str
    transcript: Transcript

    @classmethod
    def from_uid(cls, uid: str) -> 'YouTubeVideo':
        transcript = Transcript([
            Text(json.get("text"), json.get("start"))
            for json in YouTubeTranscriptApi.get_transcript(uid)
        ])
        return cls(uid, transcript)
