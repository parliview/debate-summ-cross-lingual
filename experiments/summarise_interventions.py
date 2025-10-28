import os
from utils.summariser import generate_speech_summary
from tqdm import tqdm
import json
from typing import Dict, Any, Literal, Optional, List
import argparse
import logging
import asyncio
from asyncio import Semaphore

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

INTERVENTION_PATH_FORMAT = "{debate_id}/summaries/{intervention_number}-{model}-{method}.json"

async def summarise_interventions(speech: Dict[str, Any], structured: bool, model_type: Literal["ollama", "claude", "gpt"], model_name: Optional[str]) -> str:
    """
    Summarise a speech based on a given topic. Write the summary to a file. 
    
    Args:
        speech: A dictionary containing speech data
        topic: The topic to summarise the speech on
        structured: Whether to use structured summarisation
        model_type: The type of model to use ("ollama" or "claude")
        model_name: The specific model to use. If None, uses the default model for the selected type.
 
    """
    speech_path = INTERVENTION_PATH_FORMAT.format(debate_id=speech["debate_id"], intervention_number=speech["intervention_id"], model=model_name, method="structured" if structured else "unstructured")
    speech_path = os.path.join(input_dir, speech_path)
    if os.path.exists(speech_path):
        return

    topic = speech['agenda_item']
    logger.info(f"Summarising speech from english {speech['debate_id']} {speech['intervention_id']} {speech['speaker']}")
    summary_english = await generate_speech_summary(speech, topic, structured, model_type, model_name, lang='english')
    if summary_english is None:
        logger.error(f"Failed to generate summary from english speech {speech['debate_id']} {speech['mep_id']} {speech['idx']}")
        return

    logger.info(f"Summarising speech from original {speech['debate_id']} {speech['intervention_id']} {speech['speaker']}")
    if speech['lang'] == 'EN':
        summary_original = summary_english
    else:
        summary_original = await generate_speech_summary(speech, topic, structured, model_type, model_name, lang='original')
    if summary_original is None:
        logger.error(f"Failed to generate summary from original speech {speech['debate_id']} {speech['mep_id']} {speech['idx']}")
        return

    speech['summary_english'] = summary_english
    speech['summary_original'] = summary_original
    speech['model'] = model_name
    speech['method'] = "structured" if structured else "unstructured"

    os.makedirs(os.path.dirname(speech_path), exist_ok=True)
    with open(speech_path, "w") as f:
        json.dump(speech, f)

    return speech 

def get_speeches(input_dir: str) -> List[Dict[str, Any]]:
    speeches = []
    for sub_dir in os.listdir(input_dir):
        for file in os.listdir(os.path.join(input_dir, sub_dir, 'interventions')):
            with open(os.path.join(input_dir, sub_dir, 'interventions', file), "r") as f:
                speeches.append(json.load(f))
    return speeches

async def process_speech(speech: Dict[str, Any], semaphore: Semaphore, structured: bool, model_type: Literal["ollama", "claude", "gpt"  ], model_name: Optional[str]):
    """Process a single speech with a semaphore to limit concurrent requests."""
    async with semaphore:
        # logger.info(f"Summarising speech {speech['debate_id']} {speech['mep_id']} {speech['idx']}")
        await summarise_interventions(speech, structured=structured, model_type=model_type, model_name=model_name)

async def main(structured: bool, model_type: Literal["ollama", "claude", "gpt"], model_name: Optional[str], input_dir: str, cooldown: bool = False):
    # Limit concurrent requests to avoid overwhelming the API
    semaphore = Semaphore(3)  # Adjust this number based on API limits and system resources
    
    # Create tasks for all speeches
    tasks = [
        process_speech(speech, semaphore, structured, model_type, model_name)
        for speech in get_speeches(input_dir)
    ]
    
    # Process speeches concurrently with progress bar
    with tqdm(total=len(tasks), desc="Processing speeches") as pbar:
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                await coro
            except Exception as e:
                logger.error(f"Task failed: {str(e)}")
            pbar.update(1)
            
            # Add wait time every 100 iterations
            if cooldown:
                if (i + 1) % 20 == 0:
                    await asyncio.sleep(30)  # Wait 30 seconds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', required=True, help='Input directory')
    parser.add_argument('--structured', action='store_true', help='Use structured summarization format')
    parser.add_argument('--model-type', choices=['ollama', 'claude', 'gpt'], required=True, help='Model type to use')
    parser.add_argument('--model-name', help='Specific model name to use')
    parser.add_argument('--cooldown', action='store_true', help='Cooldown after every 100 iterations')
    args = parser.parse_args()
    
    structured = args.structured
    model_type = args.model_type
    model_name = args.model_name
    input_dir = args.input_dir
    cooldown = args.cooldown

    asyncio.run(main(structured, model_type, model_name, input_dir, cooldown))