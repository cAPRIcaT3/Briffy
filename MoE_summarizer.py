#!/usr/bin/env python3
"""
YouTube Video Summarizer with Mixture of Experts (MoE) approach
Different specialized models for different video types
"""

import sys
import re
import logging
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional
import json
import os

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import yt_dlp
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoTypeClassifier:
    """Classifies video type to route to appropriate expert"""

    def __init__(self):
        self.video_types = {
            0: "news",
            1: "comedy",
            2: "reaction",
            3: "informative",
            4: "sensational",
            5: "gamer"
        }

        # Keywords for rule-based classification (fallback)
        self.keywords = {
            "news": ["breaking", "news", "report", "today", "update", "current", "politics", "world"],
            "comedy": ["funny", "comedy", "humor", "laugh", "joke", "meme", "parody", "sketch"],
            "reaction": ["reaction", "reacts", "watching", "first time", "response", "react"],
            "informative": ["tutorial", "how to", "guide", "learn", "education", "explain", "science"],
            "sensational": ["shocking", "amazing", "incredible", "unbelievable", "crazy", "wild", "insane"],
            "gamer": ["gaming", "gameplay", "game", "player", "stream", "twitch", "minecraft", "fortnite"]
        }

    def classify_by_metadata(self, title: str, description: str = "", tags: List[str] = None) -> str:
        """Rule-based classification using video metadata"""
        text = f"{title} {description} {' '.join(tags or [])}".lower()

        scores = {}
        for category, keywords in self.keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[category] = score

        # Return category with highest score, default to informative
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "informative"

    def classify_by_transcript(self, transcript: str) -> str:
        """Classify based on transcript content patterns"""
        transcript_lower = transcript.lower()

        # Simple pattern matching for transcript classification
        patterns = {
            "news": ["according to", "reports", "sources", "today", "breaking"],
            "comedy": ["haha", "lol", "funny", "joke", "hilarious"],
            "reaction": ["oh my god", "what", "no way", "that's crazy", "wait"],
            "informative": ["first", "second", "step", "now", "let's", "you need to"],
            "sensational": ["can't believe", "amazing", "incredible", "shocking"],
            "gamer": ["level", "kill", "game", "player", "spawn", "boss"]
        }

        scores = {}
        for category, pattern_words in patterns.items():
            score = sum(1 for word in pattern_words if word in transcript_lower)
            scores[category] = score

        return max(scores, key=scores.get) if max(scores.values()) > 0 else "informative"


class ExpertSummarizer:
    """Individual expert for specific video type"""

    def __init__(self, expert_type: str, model_config: Dict):
        self.expert_type = expert_type
        self.model_config = model_config
        self.summarizer = None
        self.load_model()

    def load_model(self):
        """Load the specialized model for this expert"""
        try:
            # Different models/configs for different types
            if self.expert_type == "news":
                # News summarization - focus on key facts
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",  # Good for news
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=150,
                    min_length=50
                )
            elif self.expert_type == "comedy":
                # Comedy - preserve humor and timing
                self.summarizer = pipeline(
                    "summarization",
                    model="pszemraj/long-t5-tglobal-base-16384-book-summary",
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=200,
                    min_length=80
                )
            elif self.expert_type == "reaction":
                # Reaction - focus on emotions and responses
                self.summarizer = pipeline(
                    "summarization",
                    model="pszemraj/long-t5-tglobal-base-16384-book-summary",
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=180,
                    min_length=60
                )
            elif self.expert_type == "informative":
                # Educational - structured and detailed
                self.summarizer = pipeline(
                    "summarization",
                    model="pszemraj/long-t5-tglobal-base-16384-book-summary",
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=250,
                    min_length=100
                )
            elif self.expert_type == "sensational":
                # Sensational - capture key dramatic points
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=160,
                    min_length=60
                )
            elif self.expert_type == "gamer":
                # Gaming - focus on gameplay and mechanics
                self.summarizer = pipeline(
                    "summarization",
                    model="pszemraj/long-t5-tglobal-base-16384-book-summary",
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=200,
                    min_length=80
                )

            logger.info(f"✓ Loaded {self.expert_type} expert")

        except Exception as e:
            logger.error(f"Failed to load {self.expert_type} expert: {e}")
            # Fallback to basic summarizer
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )

    def get_prompt_template(self) -> str:
        """Get specialized prompt template for this expert"""
        templates = {
            "news": "Summarize the key facts, events, and important details from this news content:",
            "comedy": "Summarize this comedy content while preserving the humor and main funny moments:",
            "reaction": "Summarize this reaction video focusing on the creator's responses and emotions:",
            "informative": "Create a structured summary of this educational content, highlighting key learning points:",
            "sensational": "Summarize this content highlighting the most dramatic and attention-grabbing elements:",
            "gamer": "Summarize this gaming content focusing on gameplay, strategy, and key moments:"
        }
        return templates.get(self.expert_type, "Summarize this content:")

    def summarize(self, text: str) -> str:
        """Generate summary using this expert"""
        try:
            # Add expert-specific preprocessing
            processed_text = self.preprocess_for_expert(text)

            # Generate summary
            result = self.summarizer(processed_text)
            if isinstance(result, list):
                return result[0]['summary_text']
            return result['summary_text']

        except Exception as e:
            logger.error(f"Expert {self.expert_type} summarization failed: {e}")
            raise

    def preprocess_for_expert(self, text: str) -> str:
        """Expert-specific text preprocessing"""
        # Truncate if too long for model
        max_length = 4000  # Adjust based on model
        if len(text) > max_length:
            text = text[:max_length]

        # Expert-specific preprocessing
        if self.expert_type == "news":
            # For news, prioritize beginning where key facts usually are
            return text[:max_length]
        elif self.expert_type == "comedy":
            # For comedy, try to preserve punchlines and key moments
            return text
        elif self.expert_type == "reaction":
            # For reactions, focus on emotional responses
            return text
        elif self.expert_type == "informative":
            # For educational, preserve structure
            return text
        elif self.expert_type == "sensational":
            # For sensational, prioritize dramatic parts
            return text
        elif self.expert_type == "gamer":
            # For gaming, focus on gameplay descriptions
            return text

        return text


class MoEYouTubeSummarizer:
    """Main MoE system that routes to appropriate experts"""

    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback or print
        self.classifier = VideoTypeClassifier()
        self.experts = {}
        self.load_experts()

    def load_experts(self):
        """Load all expert models"""
        self.progress_callback("Loading MoE experts...")

        expert_configs = {
            "news": {"model": "facebook/bart-large-cnn"},
            "comedy": {"model": "pszemraj/long-t5-tglobal-base-16384-book-summary"},
            "reaction": {"model": "pszemraj/long-t5-tglobal-base-16384-book-summary"},
            "informative": {"model": "pszemraj/long-t5-tglobal-base-16384-book-summary"},
            "sensational": {"model": "facebook/bart-large-cnn"},
            "gamer": {"model": "pszemraj/long-t5-tglobal-base-16384-book-summary"}
        }

        for expert_type, config in expert_configs.items():
            try:
                self.experts[expert_type] = ExpertSummarizer(expert_type, config)
            except Exception as e:
                logger.error(f"Failed to load {expert_type} expert: {e}")

        self.progress_callback(f"✓ Loaded {len(self.experts)} experts")

    def extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
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
        """Get video metadata"""
        try:
            ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                title = info.get('title', 'Unknown Title')
                description = info.get('description', '')
                tags = info.get('tags', [])
                duration = info.get('duration')

                if duration:
                    hours = duration // 3600
                    minutes = (duration % 3600) // 60
                    seconds = duration % 60
                    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
                else:
                    duration_str = None

                return {
                    'title': title,
                    'description': description,
                    'tags': tags,
                    'duration': duration_str
                }
        except Exception as e:
            logger.warning(f"Could not get video info: {e}")
            return {
                'title': 'Unknown Title',
                'description': '',
                'tags': [],
                'duration': None
            }

    def get_transcript(self, video_id, language='en'):
        """Get video transcript"""
        self.progress_callback("Fetching transcript...")
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try to get transcript in preferred language
            try:
                transcript = transcript_list.find_transcript([language])
            except:
                transcript = transcript_list.find_generated_transcript([language])

            transcript_data = transcript.fetch()
            formatter = TextFormatter()
            transcript_text = formatter.format_transcript(transcript_data)

            if not transcript_text or len(transcript_text.strip()) < 50:
                raise Exception("Transcript is too short or empty")

            self.progress_callback(f"✓ Transcript fetched ({len(transcript_text)} chars)")
            return transcript_text

        except Exception as e:
            logger.error(f"Transcript fetch error: {e}")
            raise Exception(f"Could not fetch transcript: {e}")

    def classify_video(self, video_info: Dict, transcript: str) -> str:
        """Classify video type to select appropriate expert"""
        self.progress_callback("Classifying video type...")

        # Try metadata-based classification first
        metadata_type = self.classifier.classify_by_metadata(
            video_info['title'],
            video_info['description'],
            video_info['tags']
        )

        # Try transcript-based classification
        transcript_type = self.classifier.classify_by_transcript(transcript)

        # Simple voting mechanism
        if metadata_type == transcript_type:
            selected_type = metadata_type
        else:
            # Default to metadata classification if they disagree
            selected_type = metadata_type

        self.progress_callback(f"✓ Video classified as: {selected_type}")
        return selected_type

    def summarize_video(self, youtube_url: str, language: str = 'en', force_expert: str = None):
        """Summarize video using MoE approach"""
        try:
            # Extract video info
            video_id = self.extract_video_id(youtube_url)
            self.progress_callback(f"✓ Video ID: {video_id}")

            video_info = self.get_video_info(video_id)
            self.progress_callback(f"✓ Video: {video_info['title']}")

            # Get transcript
            transcript = self.get_transcript(video_id, language)

            # Classify video type or use forced expert
            if force_expert and force_expert in self.experts:
                expert_type = force_expert
                self.progress_callback(f"✓ Using forced expert: {expert_type}")
            else:
                expert_type = self.classify_video(video_info, transcript)

            # Route to appropriate expert
            if expert_type not in self.experts:
                expert_type = "informative"  # Default fallback
                self.progress_callback(f"! Falling back to {expert_type} expert")

            expert = self.experts[expert_type]
            self.progress_callback(f"Generating summary with {expert_type} expert...")

            # Generate summary
            summary = expert.summarize(transcript)

            self.progress_callback("✓ Processing complete")

            return {
                'title': video_info['title'],
                'duration': video_info['duration'],
                'video_type': expert_type,
                'expert_used': expert_type,
                'transcript': transcript,
                'summary': summary,
                'url': youtube_url,
                'language': language
            }

        except Exception as e:
            error_msg = f"Error processing video: {e}"
            logger.error(error_msg)
            self.progress_callback(error_msg)
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YouTube MoE Summarizer")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--language", default="en", help="Transcript language")
    parser.add_argument("--expert", choices=["news", "comedy", "reaction", "informative", "sensational", "gamer"],
                        help="Force specific expert")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    # Initialize MoE summarizer
    summarizer = MoEYouTubeSummarizer()

    # Process video
    result = summarizer.summarize_video(args.url, args.language, args.expert)

    if result:
        print("\n" + "=" * 80)
        print(f"TITLE: {result['title']}")
        if result.get('duration'):
            print(f"DURATION: {result['duration']}")
        print(f"VIDEO TYPE: {result['video_type']}")
        print(f"EXPERT USED: {result['expert_used']}")
        print(f"LANGUAGE: {result['language']}")
        print("=" * 80)
        print(f"SUMMARY:\n{result['summary']}")
        print("=" * 80)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json.dumps(result, indent=2, ensure_ascii=False))
            print(f"✓ Results saved to {args.output}")
    else:
        print("Failed to process video")
        sys.exit(1)