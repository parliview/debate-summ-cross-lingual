from utils.prompts import SUMMARIZER_SYSTEM_PROMPT_STRUCTURED, SUMMARIZER_SYSTEM_PROMPT_UNSTRUCTURED, SUMMARIZER_USER_PROMPT
from typing import Dict, Any, Literal, Optional
import aiohttp
import asyncio
import logging
from dotenv import load_dotenv
import os
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
OLLAMA_API_TIMEOUT = aiohttp.ClientTimeout(total=float(os.getenv("OLLAMA_API_TIMEOUT", 120)))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
DEFAULT_CLAUDE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
DEFAULT_GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4-turbo-preview")

def format_speech_text(speech: Dict[str, Any], lang: str = 'english') -> str:
    """Format speech data into a consistent text format."""
    return f"""
    {speech[lang]}\n
    """



async def generate_summary_with_ollama(system_prompt: str, user_prompt: str, model: str) -> str:
    """Generate summary using Ollama API."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    
    async with aiohttp.ClientSession(timeout=OLLAMA_API_TIMEOUT) as session:
        try:
            async with session.post(OLLAMA_API_URL, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("message", {}).get("content", "")
        except asyncio.TimeoutError:
            logger.error(f"Timeout while generating summary with Ollama model: {model}")
            return None
        except Exception as e:
            logger.error(f"Error generating summary with Ollama model {model}: {str(e)}")
            return None

async def generate_summary_with_claude(system_prompt: str, user_prompt: str, model: str) -> str:
    """Generate summary using Claude API."""
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    
    try:
        message = await client.messages.create(
            model=model,
            max_tokens=8000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        logger.error(f"Error generating summary with Claude model {model}: {str(e)}")
        return None

async def generate_summary_with_gpt(system_prompt: str, user_prompt: str, model: str) -> str:
    """Generate summary using OpenAI API."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating summary with GPT model {model}: {str(e)}")
        return None

async def generate_speech_summary(
    speech: Dict[str, Any], 
    topic: str, 
    structured: bool = True,
    model_type: Literal["ollama", "claude", "gpt"] = "claude",
    model_name: Optional[str] = None,
    lang: str = 'english'
) -> str:
    """
    Generate a summary for a speech using either Ollama, Claude, or GPT.
    
    Args:
        speech: Dictionary containing speech data
        topic: Topic to focus on in the summary
        structured: Whether to use structured or unstructured summary format
        model_type: Which type of model to use ("ollama", "claude", or "gpt")
        model_name: Specific model to use. If None, uses the default model for the selected type.
                   For Ollama: e.g., "llama2", "mistral", "codellama"
                   For Claude: e.g., "claude-3-sonnet-20240229", "claude-3-opus-20240229"
                   For GPT: e.g., "gpt-4-turbo-preview", "gpt-3.5-turbo"
    
    Returns:
        str: Generated summary or None if generation fails
    """
    speech_text = format_speech_text(speech, lang)
    system_prompt = SUMMARIZER_SYSTEM_PROMPT_STRUCTURED if structured else SUMMARIZER_SYSTEM_PROMPT_UNSTRUCTURED
    user_prompt = SUMMARIZER_USER_PROMPT.format(speech_text=speech_text, topic=topic)

    # Use default model if none specified
    if model_name is None:
        if model_type == "ollama":
            model_name = DEFAULT_OLLAMA_MODEL
        elif model_type == "claude":
            model_name = DEFAULT_CLAUDE_MODEL
        else:
            model_name = DEFAULT_GPT_MODEL

    if model_type == "ollama":
        return await generate_summary_with_ollama(system_prompt, user_prompt, model_name)
    elif model_type == "claude":
        return await generate_summary_with_claude(system_prompt, user_prompt, model_name)
    else:
        return await generate_summary_with_gpt(system_prompt, user_prompt, model_name)