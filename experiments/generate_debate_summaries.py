import json
from utils.generator import generate_incremental_debate_summary, generate_debate_summary, generate_hiearchical_debate_summary, generate_zero_shot_debate_summary
from utils.formatting import parse_contributions
import os
import asyncio
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

def get_debate_title(debate_id: str) -> str:
    summary = os.listdir(os.path.join(debate_id, 'summaries'))[0]
    with open(os.path.join(debate_id, 'summaries', summary), 'r') as f:
        debate_info = json.load(f)
    return debate_info['agenda_item']

def get_contributions(debate_id: str, model_type: str, structured: bool, lang: str, zero_shot: bool) -> List[Dict[str, Any]]:
    # logger.warning('looking for contributions')
    contribution_dir = os.path.join(debate_id, 'summaries')
    contribution_files = os.listdir(contribution_dir)
    contributions = []
    for contribution_file in sorted(contribution_files):
        if model_type in contribution_file:
            # skip unstructured contributions if structured is True and vice versa
            if (not zero_shot) and (structured == contribution_file.endswith('unstructured.json')):
                continue
            with open(f'{contribution_dir}/{contribution_file}', 'r') as f:
                contribution_data = json.load(f)
                contributions.append(contribution_data)

    # logger.warning(f"Found {len(contributions)} contributions")
    return parse_contributions(contributions=contributions, lang=lang, zero_shot=zero_shot)

def get_out_id(debate_id: str,
                src_model: str,
                structured: bool,
                incremental: bool,
                grouped: bool,
                shuffle: bool,
                prompt: bool,
                hierarchical: bool,
                lang: str,
                zero_shot: bool) -> str:

    lang_format = {
        'english': 'en',
        'original': 'org'
    }

    out_id = debate_id 
    out_id += f'_{src_model}'
    if structured:
        out_id += '_structured'
    if incremental:
        out_id += '_incremental'
    if grouped:
        out_id += '_grouped'
    if shuffle:
        out_id += '_shuffled'
    if prompt:
        out_id += '_prompt'
    if hierarchical:
        out_id += '_hierarchical'
    if lang:
        out_id += f'_{lang_format[lang]}'
    if zero_shot:
        out_id += '_zero_shot'
    return out_id

async def summarise_debate( input_dir: str,
                            debate_id: str,
                            model_type: str,
                            model_name: str,
                            structured: bool,
                            incremental: bool,
                            grouped: bool,
                            shuffle: bool,
                            prompt: bool,
                            hierarchical: bool,
                            batch_size: int,
                            src_model: str,
                            overwrite: bool,
                            lang: str,
                            zero_shot: bool) -> str:
    """
    Summarise a debate.
    """
    if src_model is None:
        src_model = model_type

    debate_title = get_debate_title(os.path.join(input_dir, debate_id))

    assert not (incremental and (prompt or grouped or shuffle or hierarchical)), "Cannot have incremental with other options"
    assert not (zero_shot and (prompt or grouped or shuffle or hierarchical or structured)), "Cannot have zero-shot with other options"
    if hierarchical:
        assert structured, "Hierarchical summarisation requires structured contributions"

    out_id = get_out_id(debate_id=debate_id, src_model=src_model, structured=structured, incremental=incremental, grouped=grouped, shuffle=shuffle, prompt=prompt, hierarchical=hierarchical, lang=lang, zero_shot=zero_shot)
    #check report dir exists:
    report_dir = os.path.join(input_dir, debate_id, 'reports')
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    outpath = os.path.join(report_dir, f'{out_id}.json')
    if (os.path.exists(outpath) and not overwrite):
        return

    # get contributions
    # logger.warning('Getting contributions')
    contributions = get_contributions(debate_id=os.path.join(input_dir, debate_id), model_type=src_model, structured=structured, lang=lang, zero_shot=zero_shot)
    assert len(contributions) > 0, f'No contributions found for {debate_id}'
    # generate summary
    if incremental:
        summary = await generate_incremental_debate_summary(contributions=contributions, debate_title=debate_title, model_type=model_type, model_name=model_name, batch_size=batch_size)
    elif hierarchical:
        summary = await generate_hiearchical_debate_summary(contributions=contributions, debate_title=debate_title, model_type=model_type, model_name=model_name)
    elif zero_shot:
        summary = await generate_zero_shot_debate_summary(contributions=contributions, debate_title=debate_title, model_type=model_type, model_name=model_name)
    else:
        summary = await generate_debate_summary(contributions=contributions, debate_title=debate_title, model_type=model_type, model_name=model_name, grouped=grouped, shuffle=shuffle, prompt=prompt)

    report = {'contributions': contributions, 'summary': summary, 'src_model': src_model, 'model_type': model_type, 'model_name': model_name, 'structured': structured, 'incremental': incremental, 'hierarchical': hierarchical, 'grouped': grouped, 'shuffle': shuffle, 'prompt': prompt, 'lang': lang}
    with open(outpath, 'w') as f:
        json.dump(report, f)

    return report



async def process_debate(semaphore: asyncio.Semaphore,
                         input_dir: str,
                         debate_id: str,
                         model_type: str,
                         model_name: str,
                         structured: bool,
                         incremental: bool,
                         hierarchical: bool,
                         grouped: bool,
                         shuffle: bool,
                         prompt: bool,
                         batch_size: int,
                         src_model: str,
                         overwrite: bool,
                         zero_shot: bool):

    async with semaphore:
        try:
            await summarise_debate(input_dir=input_dir, debate_id=debate_id, model_type=model_type, model_name=model_name,
                                    structured=structured, incremental=incremental, hierarchical=hierarchical, grouped=grouped,
                                    shuffle=shuffle, prompt=prompt, batch_size=batch_size, src_model=src_model, overwrite=overwrite,
                                    lang='english', zero_shot=zero_shot)
        except Exception as e:
            logger.error(f"Error summarising debate {debate_id} in English: {e}")
        try:
            await summarise_debate(input_dir=input_dir, debate_id=debate_id, model_type=model_type, model_name=model_name,
                                    structured=structured, incremental=incremental, hierarchical=hierarchical, grouped=grouped,
                                    shuffle=shuffle, prompt=prompt, batch_size=batch_size, src_model=src_model, overwrite=overwrite,
                                    lang='original', zero_shot=zero_shot)
        except Exception as e:
            logger.error(f"Error summarising debate {debate_id} in original languages: {e}")

async def main(model_type: str, model_name: str, structured: bool, incremental: bool, hierarchical: bool, grouped: bool, shuffle: bool, prompt: bool, batch_size: int, src_model: str, cooldown: bool, overwrite: bool, zero_shot: bool):
    semaphore = asyncio.Semaphore(1)
    debate_ids = os.listdir(input_dir)
    tasks = [
        process_debate(semaphore=semaphore, input_dir=input_dir, debate_id=debate_id,
                        model_type=model_type, model_name=model_name, structured=structured,
                          incremental=incremental, grouped=grouped, shuffle=shuffle,
                          hierarchical=hierarchical, prompt=prompt, batch_size=batch_size,
                           src_model=src_model, overwrite=overwrite, zero_shot=zero_shot)
        for debate_id in debate_ids
    ]
    with tqdm(total=len(tasks), desc=f"Processing debates") as pbar:
        for i, task in enumerate(asyncio.as_completed(tasks)):
            await task
            pbar.update(1)

            # if cooldown:
                # if (i+1) % 2 == 0:
            await asyncio.sleep(10)

if __name__ == "__main__":
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, help='Input directory containing debate data')
    parser.add_argument('--debate-id', type=str, default=None, help='ID of debate to summarize')
    parser.add_argument('--model-type', type=str, default='ollama', help='Model type (ollama or claude)')
    parser.add_argument('--model-name', type=str, default='mistral', help='Name of model to use')
    parser.add_argument('--structured', action='store_true', default=False, help='Use structured contributions')
    parser.add_argument('--incremental', action='store_true', default=False, help='Generate summary incrementally')
    parser.add_argument('--hierarchical', action='store_true', default=False, help='Generate hierarchical summary')
    parser.add_argument('--grouped', action='store_true', default=False, help='Group contributions by speaker')
    parser.add_argument('--shuffle', action='store_true', default=False, help='Shuffle contributions')
    parser.add_argument('--zero-shot', action='store_true', default=False, help='Use zero-shot generation')
    parser.add_argument('--prompt', action='store_true', default=False, help='Use explicit prompt')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for incremental generation')
    parser.add_argument('--src-model', type=str, default=None, help='Source model for contributions')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing summaries')
    args = parser.parse_args()

    model_type = args.model_type
    model_name = args.model_name
    debate_id = args.debate_id
    input_dir = args.input_dir
    structured = args.structured
    incremental = args.incremental
    hierarchical = args.hierarchical
    grouped = args.grouped
    shuffle = args.shuffle
    zero_shot = args.zero_shot
    prompt = args.prompt
    batch_size = args.batch_size
    src_model = args.src_model
    overwrite = args.overwrite
    cooldown = model_type != 'ollama'

    asyncio.run(main(
        model_type=model_type, model_name=model_name,
        structured=structured, incremental=incremental,
        hierarchical=hierarchical, grouped=grouped,
        shuffle=shuffle, prompt=prompt, zero_shot=zero_shot,
        batch_size=batch_size, src_model=src_model,
        cooldown=cooldown, overwrite=overwrite
    ))
    