from fastapi import FastAPI, Query
from pydantic import BaseModel
from summarizer import YouTubeSummarizer  # your refactored logic

app = FastAPI()
summarizer = YouTubeSummarizer()

class SummaryResponse(BaseModel):
    title: str
    duration: str | None
    summary: str
    url: str
    language: str
    transcript: str

@app.get("/summarize", response_model=SummaryResponse)
def summarize(url: str = Query(...), language: str = Query("en")):
    result = summarizer.summarize_video(url, language)
    if not result:
        return {"error": "Failed to summarize"}
    return result
