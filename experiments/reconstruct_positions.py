from utils.reconstructor import reconstruct_debate
import os
import json
import asyncio
from tqdm import tqdm
import argparse
report_params = ['model_name', 'model_type', 'grouped', 'shuffle', 'prompt', 'incremental', 'hierarchical', 'src_model', 'batch_size']

async def main(model_name: str, model_type: str, input_dir: str):
    
    for debate_id in tqdm(os.listdir(input_dir)):
    # for debate_id in ['debate_1']:
        report_dir = os.path.join(input_dir, debate_id, 'reports')
        report_files = os.listdir(report_dir)
            
        out_dir = os.path.join(input_dir, debate_id, 'reconstructed_summaries', model_name)
        os.makedirs(out_dir, exist_ok=True)

        for report_file in tqdm(report_files):
            out_path = os.path.join(out_dir, report_file)
            if (os.path.exists(out_path)):
                continue
            with open(os.path.join(report_dir, report_file), 'r') as f:
                debate_report = json.load(f)
            updated_contributions = await reconstruct_debate(debate_report = debate_report, model_type = model_type, model_name = model_name)
            for contribution in updated_contributions:
                for par in report_params:
                    if par in debate_report:
                        contribution[f"generator_{par}"] = debate_report[par]
            with open(out_path, 'w') as f:
                json.dump(updated_contributions, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='mistral')
    parser.add_argument('--model-type', type=str, default='ollama')
    parser.add_argument('--input-dir', type=str, default='../debate_data')
    args = parser.parse_args()

    asyncio.run(main(args.model_name, args.model_type, args.input_dir))