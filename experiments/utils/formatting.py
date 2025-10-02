import json
from typing import Dict, Any, List, Optional
import random
import logging

logger = logging.getLogger(__name__)

REFORMAT_HEADERS = {
    'headline': 'Headline',
    'summary': 'Summary',
    'quotes': 'Quotes',
    'argSum': 'Argument Summary',
    'propSum': 'Proposal Summary',
    'positionSum': 'Position Summary',
    'issueSum': 'Issue Summary'
}

def format_contribution(contribution: Dict[str, Any], sub_heading: Optional[str] = None) -> str:
    """
    Format a contribution into a single string.
    """

    if sub_heading:
        headings = [sub_heading, 'quotes']
    else:
        headings = contribution['formatted_summary'].keys()

    output = ""
    output += f"{contribution['speaker']}:\n"
    if 'headline' in contribution['formatted_summary']:
        if contribution['formatted_summary']['headline'] == '':
            return ""
        output += f"{REFORMAT_HEADERS['headline']}: {contribution['formatted_summary']['headline']}\n"
    for heading in headings:
        if heading == 'headline':
            continue
        if heading in contribution['formatted_summary']:
            if heading in REFORMAT_HEADERS:
                output += f"{REFORMAT_HEADERS[heading]}: {contribution['formatted_summary'][heading]}\n"
            else:
                output += f"{heading}: {contribution['formatted_summary'][heading]}\n"
    return output

def format_contributions(contributions: Dict[str, List[str]], grouped: bool, shuffle: bool) -> str:
    """
    Format the a list of contributions into a single string.
    args:
        contributions: a list of contributions
        grouped: whether to group the contributions by heading
        shuffle: whether to shuffle the contributions
    returns:
        a single string of the formatted contributions
    """
    formatted_contributions = ""

    # filter out contributions that don't have a formatted summary
    contributions = [c for c in contributions if 'formatted_summary' in c]

    if grouped: 
        headings = {}
        for contribution in contributions:
            for heading in contribution['formatted_summary']:
                if heading not in headings:
                    headings[heading] = []
                headings[heading].append(f"{contribution['speaker']}: {contribution['formatted_summary'][heading]}")
        for heading in headings:
            if heading in REFORMAT_HEADERS:
                formatted_contributions += f"{REFORMAT_HEADERS[heading]}:\n"
            else:
                formatted_contributions += f"{heading}:\n"
            if shuffle:
                random.shuffle(headings[heading])
            for contribution in headings[heading]:
                formatted_contributions += f"{contribution}\n"
            formatted_contributions += f"-----------------------------------\n"
    else:
        num_speakers = len(contributions)
        speaker_ids = list(range(num_speakers))
        if shuffle:
            random.shuffle(speaker_ids)
        for speaker_id in speaker_ids:
            formatted_contributions += f"Speaker {speaker_id+1}:\n"
            formatted_contributions += format_contribution(contributions[speaker_id])
            formatted_contributions += f"-----------------------------------\n"
    return formatted_contributions

def format_contributions_identity(contributions: List[Dict[str, Any]]) -> str:
    """
    Format the a list of contributions into a single string.
    """
    formatted_contributions = ""
    for contribution in contributions:
        formatted_contributions += f"{contribution['speaker']}: {contribution['formatted_summary']}\n"
    return formatted_contributions


def _try_parse(summary):

    if isinstance(summary, dict):
        return summary

    try:
        return json.loads(summary)
    except:
        if '```json' in summary:
            summary = summary.split('```json')[1].split('```')[0]
            summary_dict = json.loads(summary)
            if 'headline' in summary_dict:
                return summary_dict
            else:
                return {}
        elif summary[-1] != '}':
            try:
                summary_dict = json.loads(summary + '}')
                if 'headline' in summary_dict:
                    return summary_dict
                else:
                    return {}
            except:
                return {}
        else:
            return {}

def parse_contributions(contributions: List[Dict[str, Any]], lang: str, zero_shot: bool) -> List[Dict[str, Any]]:
    """
    Parse the contributions into a list of dictionaries.
    """
    # unpack the contributions
    logger.info(f"Parsing {len(contributions)} contributions")
    for contribution in contributions:
        if zero_shot:
            summary = contribution[lang]
        else: 
            summary = _try_parse(contribution[f'summary_{lang}'])
            if summary == {}:
                logger.warning(f"Cannot parse summary for {contribution['debate_id']} {contribution['intervention_id']} {contribution['speaker']}")
        contribution['formatted_summary'] = summary
    return contributions
