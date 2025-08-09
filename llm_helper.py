from openai import OpenAI
import json
import os

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_prompt(prompt_file):
    """Load prompt from text file"""
    try:
        with open(f"prompts/{prompt_file}", 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Prompt file {prompt_file} not found!")
        return None

def call_gpt4o(prompt, user_content, max_tokens=2000):
    """Call GPT-4o with the given prompt and content"""
    try:
        # Initialize OpenAI client with API key
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=max_tokens,
            temperature=0.1  # Low temperature for consistent results
        )
        
        content = response.choices[0].message.content
        if content:
            return content.strip()
        else:
            print("GPT-4o returned empty content")
            return None
    
    except Exception as e:
        print(f"Error calling GPT-4o: {str(e)}")
        return None

def breakdown_question(question_content):
    """Break down question.txt using GPT-4o"""
    prompt = load_prompt("breakdown_question.txt")
    if not prompt:
        print("Could not load breakdown prompt")
        return None
    
    response = call_gpt4o(prompt, question_content)
    
    if response is None:
        print("GPT-4o call failed - no response received")
        return None
    
    try:
        # Clean the response - remove markdown code blocks if present
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]  # Remove ```json
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]  # Remove ```
        clean_response = clean_response.strip()
        
        # Try to parse as JSON
        breakdown = json.loads(clean_response)
        return breakdown
    except json.JSONDecodeError as e:
        print(f"Failed to parse GPT-4o response as JSON: {str(e)}")
        print("Raw response:", response)
        return None

def get_data_metadata(breakdown):
    """Get metadata about the data structure needed using GPT-4o"""
    prompt = load_prompt("get_metadata.txt")
    if not prompt:
        print("Could not load metadata prompt")
        return None
    
    # Create context for the LLM
    context = f"""
Data Source: {breakdown.get('data_source', {}).get('url_or_path', 'N/A')}
Data Type: {breakdown.get('data_source', {}).get('type', 'N/A')}
Description: {breakdown.get('data_source', {}).get('description', 'N/A')}

Questions to Answer:
"""
    
    for i, task in enumerate(breakdown.get('tasks', []), 1):
        context += f"{i}. {task.get('question', 'N/A')}\n"
    
    context += f"\nExpected Response Format: {breakdown.get('response_format', {}).get('type', 'N/A')}"
    context += f"\nChart Required: {breakdown.get('chart_requirements', {}).get('needed', False)}"
    
    response = call_gpt4o(prompt, context)
    
    if response is None:
        print("GPT-4o call failed - no metadata received")
        return None
    
    try:
        # Clean the response - remove markdown code blocks if present
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]  # Remove ```json
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]  # Remove ```
        clean_response = clean_response.strip()
        
        # Try to parse as JSON
        metadata = json.loads(clean_response)
        return metadata
    except json.JSONDecodeError as e:
        print(f"Failed to parse metadata response as JSON: {str(e)}")
        print("Raw response:", response)
        return None

def extract_data(breakdown, metadata):
    """Generate and execute data extraction script using GPT-4o"""
    prompt = load_prompt("extract_data.txt")
    if not prompt:
        print("Could not load extraction prompt")
        return None
    
    # Create context for the LLM
    context = f"""
Data Source: {breakdown.get('data_source', {}).get('url_or_path', 'N/A')}
Data Type: {breakdown.get('data_source', {}).get('type', 'N/A')}

Target Data Location: {metadata.get('target_data', {}).get('location', 'N/A')}
Table Identifier: {metadata.get('target_data', {}).get('table_identifier', 'N/A')}
Description: {metadata.get('target_data', {}).get('description', 'N/A')}

Extraction Method: {metadata.get('extraction_method', {}).get('approach', 'N/A')}
Tools Needed: {', '.join(metadata.get('extraction_method', {}).get('tools_needed', []))}
Specific Instructions: {metadata.get('extraction_method', {}).get('specific_instructions', 'N/A')}

Required Columns:
"""
    
    for col in metadata.get('required_columns', []):
        context += f"- {col.get('column_name', 'N/A')} ({col.get('data_type', 'N/A')}): {col.get('purpose', 'N/A')}\n"
    
    context += f"\nCleaning Requirements: {', '.join(metadata.get('data_cleaning_needs', []))}"
    
    response = call_gpt4o(prompt, context, max_tokens=3000)
    
    if response is None:
        print("GPT-4o call failed - no extraction script received")
        return None
    
    # Clean the response - remove any markdown code blocks if present
    clean_script = response.strip()
    if clean_script.startswith("```python"):
        clean_script = clean_script[9:]  # Remove ```python
    elif clean_script.startswith("```"):
        clean_script = clean_script[3:]  # Remove ```
    if clean_script.endswith("```"):
        clean_script = clean_script[:-3]  # Remove ```
    clean_script = clean_script.strip()
    
    # Execute with retry mechanism
    return execute_script_with_retry(clean_script, max_retries=3, context=context, script_type="extraction")

def clean_data(breakdown, metadata, extracted_data):
    """Generate and execute data cleaning script using GPT-4o"""
    prompt = load_prompt("clean_data.txt")
    if not prompt:
        print("Could not load cleaning prompt")
        return None
    
    # Create context for the LLM
    context = f"""
Raw Data Info:
- Shape: {extracted_data.shape}
- Columns: {list(extracted_data.columns)}
- Data Types: {extracted_data.dtypes.to_dict()}

Column Purposes (from metadata):
"""
    
    for col_info in metadata.get('required_columns', []):
        context += f"- {col_info.get('column_name', 'N/A')}: {col_info.get('purpose', 'N/A')} (expected: {col_info.get('data_type', 'N/A')})\n"
    
    context += f"\nQuestions Context (for type inference):\n"
    for i, task in enumerate(breakdown.get('tasks', []), 1):
        context += f"{i}. {task.get('question', 'N/A')} (expects: {task.get('expected_output', 'N/A')})\n"
    
    context += f"\nSample Data (first 5 rows):\n{extracted_data.head().to_string()}"
    
    response = call_gpt4o(prompt, context, max_tokens=3000)
    
    if response is None:
        print("GPT-4o call failed - no cleaning script received")
        return None
    
    # Clean the response - remove any markdown code blocks if present
    clean_script = response.strip()
    if clean_script.startswith("```python"):
        clean_script = clean_script[9:]  # Remove ```python
    elif clean_script.startswith("```"):
        clean_script = clean_script[3:]  # Remove ```
    if clean_script.endswith("```"):
        clean_script = clean_script[:-3]  # Remove ```
    clean_script = clean_script.strip()
    
    # Execute with retry mechanism
    cleaning_context = {'extracted_data': extracted_data, 'context': context}
    return execute_script_with_retry(clean_script, max_retries=3, context=cleaning_context, script_type="cleaning")

def fix_script_error(original_script, error_message, context):
    """Ask LLM to fix a script error"""
    prompt = load_prompt("fix_script_error.txt")
    if not prompt:
        print("Could not load error fixing prompt")
        return None
    
    # Create context for error fixing
    fix_context = f"""
ORIGINAL SCRIPT:
{original_script}

ERROR MESSAGE:
{error_message}

CONTEXT:
{context}

Please fix the script to resolve this specific error while maintaining the original functionality.
"""
    
    response = call_gpt4o(prompt, fix_context, max_tokens=3000)
    
    if response is None:
        return None
    
    # Clean the response
    clean_script = response.strip()
    if clean_script.startswith("```python"):
        clean_script = clean_script[9:]
    elif clean_script.startswith("```"):
        clean_script = clean_script[3:]
    if clean_script.endswith("```"):
        clean_script = clean_script[:-3]
    clean_script = clean_script.strip()
    
    return clean_script

def execute_script_with_retry(script, max_retries=3, context="", script_type="extraction"):
    """Execute script with retry mechanism and error fixing"""
    
    for attempt in range(max_retries):
        try:
            # Create a namespace with required imports
            import pandas as pd
            import requests
            from bs4 import BeautifulSoup
            import re
            import json
            import chardet
            import numpy as np
            from datetime import datetime
            try:
                import duckdb
            except ImportError:
                duckdb = None
            try:
                import pdfplumber
            except ImportError:
                pdfplumber = None
            try:
                import openpyxl
            except ImportError:
                openpyxl = None
            
            script_globals = {
                'pd': pd,
                'pandas': pd,
                'np': np,
                'numpy': np,
                'requests': requests,
                'BeautifulSoup': BeautifulSoup,
                're': re,
                'json': json,
                'chardet': chardet,
                'duckdb': duckdb,
                'pdfplumber': pdfplumber,
                'openpyxl': openpyxl,
                'datetime': datetime,
                '__builtins__': __builtins__
            }
            script_locals = {}
            
            # For cleaning scripts, add the extracted_data to the namespace
            if script_type == "cleaning" and isinstance(context, dict) and 'extracted_data' in context:
                script_globals['extracted_data'] = context['extracted_data']
            
            # For answer/chart scripts, add the cleaned_data to the namespace
            if script_type in ["answers", "charts"] and isinstance(context, dict) and 'cleaned_data' in context:
                script_globals['cleaned_data'] = context['cleaned_data']
            
            # Execute the script
            exec(script, script_globals, script_locals)
            
            # Get the result based on script type
            if script_type == "extraction":
                result_data = script_locals.get('extracted_data', None)
            elif script_type == "cleaning":
                result_data = script_locals.get('cleaned_data', None)
            elif script_type == "answers":
                result_data = script_locals.get('answers', None)
            elif script_type == "charts":
                result_data = script_locals.get('chart_base64', None)
            else:
                result_data = script_locals.get('extracted_data', None) or script_locals.get('cleaned_data', None) or script_locals.get('answers', None)
            
            if result_data is not None:
                print(f"✅ Script executed successfully on attempt {attempt + 1}")
                return {
                    'script': script,
                    'data': result_data,
                    'success': True,
                    'attempts': attempt + 1
                }
            else:
                raise Exception(f"Script executed but no data was produced ({script_type}_data is None)")
        
        except Exception as e:
            error_message = str(e)
            print(f"❌ Attempt {attempt + 1}/{max_retries} failed: {error_message}")
            
            if attempt < max_retries - 1:  # Not the last attempt
                print(f"🔧 Asking LLM to fix the error...")
                fixed_script = fix_script_error(script, error_message, str(context))
                
                if fixed_script:
                    script = fixed_script  # Use the fixed script for next attempt
                    print(f"🔄 Received fixed script, retrying...")
                else:
                    print(f"❌ Could not get fixed script from LLM")
            else:
                print(f"❌ All {max_retries} attempts failed")
    
    # All attempts failed
    return {
        'script': script,
        'data': None,
        'success': False,
        'error': error_message,
        'attempts': max_retries
    }

def generate_answers(breakdown, cleaned_data):
    """Generate answers to questions using cleaned data"""
    prompt = load_prompt("generate_answers.txt")
    if not prompt:
        print("Could not load answer generation prompt")
        return None
    
    # Create context for the LLM
    context = f"""
Questions to Answer:
"""
    
    for i, task in enumerate(breakdown.get('tasks', []), 1):
        if task.get('type') != 'chart':  # Skip chart questions for now
            context += f"{i}. {task.get('question', 'N/A')} (expects: {task.get('expected_output', 'N/A')})\n"
    
    context += f"""
Response Format: {breakdown.get('response_format', {}).get('type', 'N/A')}
Format Description: {breakdown.get('response_format', {}).get('description', 'N/A')}

Data Sample (first 15 rows):
{cleaned_data.head(15).to_string()}

Data Info:
- Shape: {cleaned_data.shape}
- Columns: {list(cleaned_data.columns)}
- Data Types: {dict(cleaned_data.dtypes)}
"""
    
    response = call_gpt4o(prompt, context, max_tokens=3000)
    
    if response is None:
        print("GPT-4o call failed - no answer script received")
        return None
    
    # Clean the response
    clean_script = response.strip()
    if clean_script.startswith("```python"):
        clean_script = clean_script[9:]
    elif clean_script.startswith("```"):
        clean_script = clean_script[3:]
    if clean_script.endswith("```"):
        clean_script = clean_script[:-3]
    clean_script = clean_script.strip()
    
    # Execute with retry mechanism
    answer_context = {'cleaned_data': cleaned_data, 'context': context}
    return execute_script_with_retry(clean_script, max_retries=3, context=answer_context, script_type="answers")

def generate_charts(breakdown, cleaned_data):
    """Generate charts based on requirements"""
    chart_requirements = breakdown.get('chart_requirements', {})
    
    if not chart_requirements.get('needed', False):
        return None
    
    prompt = load_prompt("generate_charts.txt")
    if not prompt:
        print("Could not load chart generation prompt")
        return None
    
    # Create context for the LLM
    context = f"""
Chart Requirements:
- Type: {chart_requirements.get('type', 'N/A')}
- Format: {chart_requirements.get('format', 'N/A')}
- Details: {chart_requirements.get('details', 'N/A')}

Chart Questions:
"""
    
    for i, task in enumerate(breakdown.get('tasks', []), 1):
        if task.get('type') == 'chart':
            context += f"{i}. {task.get('question', 'N/A')}\n"
    
    context += f"""
Data Sample (first 15 rows):
{cleaned_data.head(15).to_string()}

Data Info:
- Shape: {cleaned_data.shape}
- Columns: {list(cleaned_data.columns)}
- Data Types: {dict(cleaned_data.dtypes)}
"""
    
    response = call_gpt4o(prompt, context, max_tokens=3000)
    
    if response is None:
        print("GPT-4o call failed - no chart script received")
        return None
    
    # Clean the response
    clean_script = response.strip()
    if clean_script.startswith("```python"):
        clean_script = clean_script[9:]
    elif clean_script.startswith("```"):
        clean_script = clean_script[3:]
    if clean_script.endswith("```"):
        clean_script = clean_script[:-3]
    clean_script = clean_script.strip()
    
    # Execute with retry mechanism
    chart_context = {'cleaned_data': cleaned_data, 'context': context}
    return execute_script_with_retry(clean_script, max_retries=3, context=chart_context, script_type="charts")
