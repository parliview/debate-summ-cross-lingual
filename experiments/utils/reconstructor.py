from typing import Literal, Optional, Dict, Any

from utils.summariser import generate_summary_with_ollama, generate_summary_with_claude
from utils.prompts import (
    RECONSTRUCTOR_SYSTEM_PROMPT_STRUCTURED, 
    RECONSTRUCTOR_USER_PROMPT_STRUCTURED, 
    RECONSTRUCTOR_SYSTEM_PROMPT_UNSTRUCTURED, 
    RECONSTRUCTOR_USER_PROMPT_UNSTRUCTURED
)


async def reconstruct_debate(debate_report: Dict[str ,Any], model_type: Literal["ollama", "claude"], model_name: Optional[str]) -> Dict[str, Any]:
    """
    Reconstruct the debate from the report.
    """
    contributions = debate_report['contributions']
    updated_contributions = []
    for contribution in contributions:
        updated_contributions.append(await reconstruct_speaker_position(debate_report, contribution, model_type, model_name))

    return updated_contributions

async def reconstruct_speaker_position(debate_report: Dict[str ,Any], contribution: Dict[str, Any], model_type: Literal["ollama", "claude"], model_name: Optional[str]) -> str:
    """
    Reconstruct the position of a speaker in a debate.
    """
    debate_summary = debate_report['summary']
    speaker_name = contribution['speaker']
    structured = debate_report['structured']

    system_prompt = RECONSTRUCTOR_SYSTEM_PROMPT_STRUCTURED if structured else RECONSTRUCTOR_SYSTEM_PROMPT_UNSTRUCTURED
    user_prompt = RECONSTRUCTOR_USER_PROMPT_STRUCTURED if structured else RECONSTRUCTOR_USER_PROMPT_UNSTRUCTURED
    user_prompt = user_prompt.format(debate_summary=debate_summary, speaker_name=speaker_name)

    if model_type == "ollama":
        reconstructed_position = await generate_summary_with_ollama(system_prompt, user_prompt, model_name)
    else:
        reconstructed_position = await generate_summary_with_claude(system_prompt, user_prompt, model_name)

    contribution['reconstructed_position'] = reconstructed_position

    return contribution

    