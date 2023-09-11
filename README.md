# YouTube-LLM

A chat bot that can answer questions given a YouTube video.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py -- --youtube_uid=Mde2q7GFCrw # https://www.youtube.com/watch?v=Mde2q7GFCrw
```

The `--youtube_uid` flag need to be passed only once, as the program will create a vector database containing data for the given video.
