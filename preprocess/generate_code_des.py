import os
import json
from openai import OpenAI
import time
import re
from difflib import SequenceMatcher

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Template for the prompt to the OpenAI model
PROMPT_TEMPLATE = """You are a clinical terminologist with expertise in ICD-10 coding. 
When provided with ICD-10 code titles, generate for EACH code:
1. Comprehensive description covering:
   - Code definition and clinical context
   - Primary indications/pathologies
   - Procedural methods (if applicable)
2. Comorbidity information including:
   - Top 1-3 complications and underlying causes
   - Pathophysiological mechanisms linking complications

STRICT OUTPUT FORMAT for EACH CODE:
[ICD-10 code title]: [The code title provided in the input]
[comprehensive description]: [A comprehensive paragraph covering the code's definition, clinical context, primary indications/pathologies, and applicable procedural methods, written in a clinical tone.];
[comorbidity information]: [A detailed paragraph describing the top 1-3 complications, underlying causes, and the pathophysiological mechanisms linking them.];

=====

Input code titles:
{code_titles}
"""

BATCH_SIZE = 10  # Batch size for processing ICD-10 code titles

def generate_batch_descriptions(code_titles):
    """Generate descriptions for a batch of ICD-10 code titles."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a medical coding specialist."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(code_titles="\n".join(code_titles))}
        ],
        temperature=0.2,
        max_tokens=5000  # Adjusted according to batch size
    )
    return parse_batch_response(response.choices[0].message.content, code_titles)

def parse_batch_response(response_text, original_titles):
    """Improved parsing function for batch responses."""
    # Preprocess: normalize line breaks and remove extra blank lines
    cleaned_text = response_text.replace("\r\n", "\n").strip()
    
    # Key modification: split code blocks using positive lookahead
    code_blocks = re.split(r"(?=\n\[ICD-10 code title\]:)", cleaned_text)
    code_blocks = [b.strip() for b in code_blocks if b.strip()]

    result_map = {}
    
    for block in code_blocks:
        try:
            # Extract the title
            title_match = re.search(
                r"^\[ICD-10 code title\]:\s*(.+?)\n", 
                block, 
                re.IGNORECASE | re.MULTILINE
            )
            if not title_match:
                continue
                
            title = title_match.group(1).strip().lower()

            # Extract the comprehensive description
            desc_match = re.search(
                r"\[comprehensive description\]:\s*(.*?)(?=\n\[comorbidity|$)", 
                block, 
                re.DOTALL | re.IGNORECASE
            )
            comp_desc = desc_match.group(1).strip() if desc_match else None

            # Extract the comorbidity information
            comorb_match = re.search(
                r"\[comorbidity information\]:\s*(.*?)(?=\n\[ICD-10|\n===|$)", 
                block, 
                re.DOTALL | re.IGNORECASE
            )
            comorbidity = comorb_match.group(1).strip() if comorb_match else None

            # Check the validity of the results
            if not all([title, comp_desc, comorbidity]):
                continue

            # Clean up extra spaces and symbols in the content
            comp_desc = re.sub(r"\s+", " ", comp_desc).replace(";", ".") if comp_desc else None
            comorbidity = re.sub(r"\s+", " ", comorbidity).replace(";", ".") if comorbidity else None

            result_map[title] = (comp_desc, comorbidity)

        except Exception as e:
            print(f"Failed to parse block: {str(e)}")
            continue

    # Create an ordered result list, handle case differences
    ordered_results = []
    for orig_title in original_titles:
        clean_orig = orig_title.strip().lower()
        ordered_results.append(result_map.get(clean_orig, (None, None)))

    return ordered_results

def process_icd_json(input_path, output_path):
    with open(input_path, 'r') as f:
        icd_data = json.load(f)
    
    items = list(icd_data.items())
    total = len(items)
    processed = 0
    results = []
    temp_prefix = os.path.splitext(output_path)[0]
    
    # Resume progress
    try:
        temp_files = [f for f in os.listdir() if f.startswith(f"{temp_prefix}_batch_")]
        if temp_files:
            last_file = sorted(temp_files)[-1]
            with open(last_file, 'r') as f:
                results = json.load(f)
            processed = len(results)
            print(f"Resumed progress from batch {processed}")
    except Exception as e:
        print(f"Failed to resume progress: {e}")

    # Main loop for batch processing
    while processed < total:
        batch_items = items[processed:processed+BATCH_SIZE]
        batch_titles = [
            item[1][0] if isinstance(item[1], list) else item[1]
            for item in batch_items
        ]
        batch_codes = [item[0] for item in batch_items]
        batch_results = generate_batch_descriptions(batch_titles)
        
        # Combine results
        for code, title, result in zip(batch_codes, batch_titles, batch_results):
            comp_desc, comorbidity = result if result else (None, None)
            results.append({
                "ICD10_CODE": code,
                "LONG_TITLE": title,
                "COMPREHENSIVE_DESC": comp_desc,
                "COMORBIDITY_INFO": comorbidity
            })
        
        processed += len(batch_items)
        print(f"\rProcessing progress: {processed}/{total} ({processed/total:.1%})", end="", flush=True)
        
        # Save checkpoint
        if processed % (BATCH_SIZE*50) == 0:
            temp_file = f"{temp_prefix}_batch_{processed}.json"
            with open(temp_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nCheckpoint saved to {temp_file}")
        
        time.sleep(1.2)  # Comply with API rate limit

    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Clean up temporary files
    for f in [x for x in os.listdir() if x.startswith(f"{temp_prefix}_batch_")]:
        os.remove(f)
    
    print(f"\nProcessing completed! Results saved to {output_path}")

if __name__ == "__main__":
    process_icd_json(
        input_path="icd_mimic4_10.json",
        output_path="enhanced_icd_mimic4_icd10.json"
    )