#!/usr/bin/env python3
"""
YouTube Video Summarizer using YouTube Transcript API
Refactored for API use with parallel processing and Long-T5 summarizer
"""

import sys
import re
import logging
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import yt_dlp
from transformers import pipeline
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeSummarizer:
    def __init__(self, max_chunk_length=4096, progress_callback=None):
        self.progress_callback = progress_callback or print
        self.max_chunk_length = max_chunk_length

        # Load Long-T5 summarization model
        self.summarizer = pipeline(
            "summarization",
            model="pszemraj/long-t5-tglobal-base-16384-book-summary",
            device=0 if torch.cuda.is_available() else -1,
            model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else {}
        )
        self.progress_callback("\u2713 Long-T5 summarizer loaded")

    def extract_video_id(self, url):
        parsed_url = urlparse(url)
        if parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            elif parsed_url.path.startswith('/embed/') or parsed_url.path.startswith('/v/'):
                return parsed_url.path.split('/')[2]
        raise ValueError("Invalid YouTube URL")

    def get_video_info(self, video_id):
        try:
            ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                title = info.get('title', 'Unknown Title')
                duration = info.get('duration')
                if duration:
                    hours = duration // 3600
                    minutes = (duration % 3600) // 60
                    seconds = duration % 60
                    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
                else:
                    duration_str = None
                return title, duration_str
        except Exception as e:
            logger.warning(f"Could not get video info: {e}")
            return "Unknown Title", None

    def get_transcript(self, video_id, language='en'):
        self.progress_callback("Fetching transcript...")
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            if language == 'auto':
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                    language = 'en'
                except:
                    try:
                        transcript = transcript_list.find_manually_created_transcript(['en'])
                        language = 'en'
                    except:
                        transcript = transcript_list._manually_created_transcripts[0] \
                            if transcript_list._manually_created_transcripts else \
                            transcript_list._generated_transcripts[0]
                        language = transcript.language_code
            else:
                try:
                    transcript = transcript_list.find_transcript([language])
                except:
                    transcript = transcript_list.find_generated_transcript([language])

            transcript_data = transcript.fetch()
            formatter = TextFormatter()
            transcript_text = formatter.format_transcript(transcript_data)

            if not transcript_text or len(transcript_text.strip()) < 50:
                raise Exception("Transcript is too short or empty")

            self.progress_callback(f"\u2713 Transcript fetched ({len(transcript_text)} chars, language: {language})")
            return transcript_text, language
        except Exception as e:
            logger.error(f"Transcript fetch error: {e}")
            raise Exception(f"Could not fetch transcript: {e}")

    def chunk_text(self, text):
        if not text or len(text.strip()) < 50:
            raise Exception("Text too short for summarization")

        sentences = re.split(r'[.!?]+', text)
        chunks, current_chunk = [], ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) > self.max_chunk_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    chunks.append(sentence)
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        chunks = [chunk for chunk in chunks if len(chunk) > 50]

        if not chunks:
            raise Exception("No valid chunks created from text")

        return chunks

    def summarize_text(self, text):
        self.progress_callback("Generating summary...")
        try:
            chunks = self.chunk_text(text)
            self.progress_callback(f"Processing {len(chunks)} chunks...")

            def summarize_chunk(i_chunk):
                i, chunk = i_chunk
                self.progress_callback(f"Summarizing chunk {i + 1}/{len(chunks)}")
                try:
                    return self.summarizer(chunk)[0]['summary_text']
                except Exception as e:
                    logger.warning(f"Failed to summarize chunk {i + 1}: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(summarize_chunk, enumerate(chunks)))

            summaries = [r for r in results if r]
            if not summaries:
                raise Exception("No chunks could be summarized")

            combined = " ".join(summaries)
            if len(combined) > self.max_chunk_length:
                self.progress_callback("Creating final summary...")
                try:
                    return self.summarizer(combined[:self.max_chunk_length])[0]['summary_text']
                except Exception as e:
                    logger.warning(f"Failed to create final summary: {e}")
            return combined
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            raise Exception(f"Failed to summarize text: {e}")

    def summarize_video(self, youtube_url, language='en'):
        try:
            video_id = self.extract_video_id(youtube_url)
            self.progress_callback(f"\u2713 Video ID: {video_id}")
            title, duration = self.get_video_info(video_id)
            self.progress_callback(f"\u2713 Video: {title}")
            transcript, detected_language = self.get_transcript(video_id, language)
            summary = self.summarize_text(transcript)
            self.progress_callback("\u2713 Processing complete")
            return {
                'title': title,
                'duration': duration,
                'transcript': transcript,
                'summary': summary,
                'url': youtube_url,
                'language': detected_language
            }
        except Exception as e:
            error_msg = f"Error processing video: {e}"
            logger.error(error_msg)
            self.progress_callback(error_msg)
            return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize YouTube videos using transcript API")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--language", default="en", help="Transcript language (en, auto, etc.)")
    parser.add_argument("--output", help="Output file path (optional)")
    args = parser.parse_args()

    summarizer = YouTubeSummarizer()
    result = summarizer.summarize_video(args.url, args.language)

    if result:
        print("\n" + "=" * 60)
        print(f"TITLE: {result['title']}")
        if result.get('duration'):
            print(f"DURATION: {result['duration']}")
        print(f"LANGUAGE: {result['language']}")
        print("=" * 60)
        print(f"SUMMARY:\n{result['summary']}")
        print("=" * 60)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"Title: {result['title']}\n")
                f.write(f"URL: {result['url']}\n")
                if result.get('duration'):
                    f.write(f"Duration: {result['duration']}\n")
                f.write(f"Language: {result['language']}\n\n")
                f.write(f"Summary:\n{result['summary']}\n\n")
                f.write(f"Full Transcript:\n{result['transcript']}")
            print(f"\u2713 Saved to {args.output}")
    else:
        print("Failed to process video")
        sys.exit(1)