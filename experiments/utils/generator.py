from utils.prompts import (
    GENERATOR_SYSTEM_PROMPT, 
    GENERATOR_USER_PROMPT, 
    INCREMENTAL_GENERATOR_SYSTEM_PROMPT, 
    INCREMENTAL_GENERATOR_USER_PROMPT, 
    EMPTY_SUMMARY,
    EXPLICIT_PROMPT,
    HIERARCHICAL_GENERATOR_HEADING_SYSTEM_PROMPT,
    HIERARCHICAL_GENERATOR_HEADING_USER_PROMPT,
    HIERARCHICAL_GENERATOR_DEBATE_SYSTEM_PROMPT,
    HIERARCHICAL_GENERATOR_DEBATE_USER_PROMPT
)
from utils.summariser import generate_summary_with_ollama, generate_summary_with_claude, generate_summary_with_gpt
from utils.formatting import format_contributions, format_contribution, REFORMAT_HEADERS, format_contributions_identity
import logging
from typing import Literal, Optional, List, Dict, Any
import aiohttp
import asyncio
import os
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
logger = logging.getLogger(__name__)
load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
OLLAMA_API_TIMEOUT = aiohttp.ClientTimeout(total=float(os.getenv("OLLAMA_API_TIMEOUT", 600)))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
DEFAULT_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")


async def generate_hiearchical_debate_summary(debate_title: str, contributions: List[Dict[str, Any]], model_type: Literal["ollama", "claude", "gpt"], model_name: Optional[str]) -> str:
    """
    Generate a hierarchical summary of a parliamentary debate.
    """
    system_prompt_heading = HIERARCHICAL_GENERATOR_HEADING_SYSTEM_PROMPT
    user_prompt_heading = HIERARCHICAL_GENERATOR_HEADING_USER_PROMPT
    system_prompt_debate = HIERARCHICAL_GENERATOR_DEBATE_SYSTEM_PROMPT
    user_prompt_debate = HIERARCHICAL_GENERATOR_DEBATE_USER_PROMPT

    batches = {} 
    for heading in ['issueSum', 'positionSum', 'argSum', 'propSum']:
        import random
        shuffled_contributions = contributions.copy()
        random.shuffle(shuffled_contributions)
        for contribution in shuffled_contributions:
            if heading not in batches:
                batches[heading] = "Speaker " + REFORMAT_HEADERS[heading].split('.')[0] + ":\n"
            batches[heading] += format_contribution(contribution, heading)


    tasks = []
    for heading in batches:
        formatted_contributions = batches[heading]
        heading_system_prompt = system_prompt_heading.format(heading=REFORMAT_HEADERS[heading])
        heading_user_prompt = user_prompt_heading.format(debate_title=debate_title, heading=REFORMAT_HEADERS[heading], contributions=formatted_contributions)
            
        if model_type == "ollama":
            task = generate_summary_with_ollama(heading_system_prompt, heading_user_prompt, model_name)
        elif model_type == "gpt":
            task = generate_summary_with_gpt(heading_system_prompt, heading_user_prompt, model_name)
        else:
            task = generate_summary_with_claude(heading_system_prompt, heading_user_prompt, model_name)
        tasks.append(task)
    
    if len(contributions) > 30:
        summaries = []
        for task in tasks:
            summary = await task
            summaries.append(summary)
            await asyncio.sleep(30)
    else:
        summaries = await asyncio.gather(*tasks)

    # summaries = ['' if not summary else summary for summary in summaries]
    formatted_summaries = "\n".join(summaries)
    user_prompt_debate = user_prompt_debate.format(debate_title=debate_title, summaries=formatted_summaries)
    if model_type == "ollama":
        return await generate_summary_with_ollama(system_prompt_debate, user_prompt_debate, model_name)
    elif model_type == "gpt":
        return await generate_summary_with_gpt(system_prompt_debate, user_prompt_debate, model_name)
    else:
        return await generate_summary_with_claude(system_prompt_debate, user_prompt_debate, model_name)

async def generate_incremental_debate_summary(debate_title: str, contributions: List[Dict[str, Any]], model_type: Literal["ollama", "claude", "gpt"], model_name: Optional[str], batch_size: int = 1) -> str:
    """
    Generate an incremental summary of a parliamentary debate.
    """
    system_prompt = INCREMENTAL_GENERATOR_SYSTEM_PROMPT
    
    summary = EMPTY_SUMMARY
    for i in range(0, len(contributions), batch_size):
        batch_contributions = contributions[i:i+batch_size]
        formatted_contributions = format_contributions(batch_contributions, grouped=False, shuffle=False)
        user_prompt = INCREMENTAL_GENERATOR_USER_PROMPT.format(debate_title=debate_title, contribution=formatted_contributions, current_summary=summary)
        if model_type == "ollama":
            summary = await generate_summary_with_ollama(system_prompt, user_prompt, model_name)
        elif model_type == "gpt":
            summary = await generate_summary_with_gpt(system_prompt, user_prompt, model_name)
        else:
            summary = await generate_summary_with_claude(system_prompt, user_prompt, model_name)

    return summary

async def generate_debate_summary(debate_title: str, contributions: List[Dict[str, Any]], model_type: Literal["ollama", "claude", "gpt"], model_name: Optional[str], grouped: bool, shuffle: bool, prompt: bool) -> str:
    """
    Generate a comprehensive summary of a parliamentary debate.
    """
    formatted_contributions = format_contributions(contributions, grouped, shuffle)
    system_prompt = GENERATOR_SYSTEM_PROMPT + EXPLICIT_PROMPT if prompt else GENERATOR_SYSTEM_PROMPT
    user_prompt = GENERATOR_USER_PROMPT.format(debate_title=debate_title, contributions=formatted_contributions)

    if model_name is None:
        model_name = DEFAULT_OLLAMA_MODEL if model_type == "ollama" else DEFAULT_CLAUDE_MODEL

    if model_type == "ollama":
        return await generate_summary_with_ollama(system_prompt, user_prompt, model_name)
    elif model_type == "gpt":
        return await generate_summary_with_gpt(system_prompt, user_prompt, model_name)
    else:
        return await generate_summary_with_claude(system_prompt, user_prompt, model_name)

async def generate_zero_shot_debate_summary(debate_title: str, contributions: List[Dict[str, Any]], model_type: Literal["ollama", "claude", "gpt"], model_name: Optional[str]) -> str:
    """
    Generate a zero-shot summary of a parliamentary debate.
    """
    formatted_contributions = format_contributions_identity(contributions)
    system_prompt = GENERATOR_SYSTEM_PROMPT
    user_prompt = GENERATOR_USER_PROMPT.format(debate_title=debate_title, contributions=contributions)
    
    if model_type == "ollama":
        return await generate_summary_with_ollama(system_prompt, user_prompt, model_name)
    elif model_type == "gpt":
        return await generate_summary_with_gpt(system_prompt, user_prompt, model_name)
    else:
        return await generate_summary_with_claude(system_prompt, user_prompt, model_name)


