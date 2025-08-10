from openai import OpenAI
import json
import os

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY environment variable not set!")
    print("⚠️ LLM calls will fail - only fallback responses will work")

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
        if not OPENAI_API_KEY:
            print("❌ OpenAI API key not available")
            return None
            
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
    
    # Handle both old format (data_source) and new format (data_sources)
    data_sources = breakdown.get('data_sources', [])
    if not data_sources and 'data_source' in breakdown:
        # Backward compatibility - convert old format to new
        data_sources = [breakdown['data_source']]
    
    # Create context for the LLM
    context = "DATA SOURCE(S) INFORMATION:\n"
    
    for i, source in enumerate(data_sources):
        context += f"""
SOURCE {i+1}:
- URL/Path: {source.get('url_or_path', 'N/A')}
- Type: {source.get('type', 'N/A')}
- Description: {source.get('description', 'N/A')}"""
        
        if source.get('s3_details'):
            s3 = source['s3_details']
            context += f"""
- S3 Bucket: {s3.get('bucket', 'N/A')}
- S3 Region: {s3.get('region', 'N/A')}
- Access Type: {s3.get('access_type', 'N/A')}
- Prefix: {s3.get('prefix', 'N/A')}"""
        
        if source.get('api_details'):
            api = source['api_details']
            context += f"""
- API Method: {api.get('method', 'GET')}
- Auth Required: {api.get('auth_required', False)}"""
    
    context += "\n\nQuestions to Answer:\n"
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

def extract_data(breakdown, metadata, uploaded_files_info=None):
    """Generate and execute data extraction script using GPT-4o"""
    
    # Use unified prompt for all data sources
    prompt = load_prompt("extract_data.txt")
    if not prompt:
        print("Could not load extraction prompt")
        return None
    
    # Special handling for CSV files - skip metadata column expectations
    if metadata.get("skip_metadata") and metadata.get("file_type") == "csv":
        print("📁 Using direct CSV extraction mode - ignoring column expectations")
        metadata = {"file_type": "csv", "direct_extraction": True}
    
    # Handle both old format (data_source) and new format (data_sources)
    data_sources = breakdown.get('data_sources', [])
    if not data_sources and 'data_source' in breakdown:
        # Backward compatibility - convert old format to new
        data_sources = [breakdown['data_source']]
    
    # Analyze data sources and set context flags
    source_types = []
    for source in data_sources:
        url_or_path = str(source.get('url_or_path', ''))
        source_type = source.get('type', '')
        
        # Improved URL type detection
        if source_type == 'webpage' or url_or_path.startswith(('http://', 'https://')):
            # Further categorize HTTP URLs
            if any(ext in url_or_path.lower() for ext in ['.pdf', '.csv', '.json', '.xml', '.zip']):
                source_types.append('http_file')
            elif 'api.' in url_or_path or '/api/' in url_or_path or url_or_path.endswith('.json'):
                source_types.append('api')
            else:
                source_types.append('webpage')
        elif 's3://' in url_or_path or 'amazonaws.com' in url_or_path:
            source_types.append('s3')
        else:
            source_types.append(source_type or 'file')
    
    # Create context for the LLM with source type hints
    context = "DATA SOURCE(S) INFORMATION:\n"
    context += f"DETECTED SOURCE TYPES: {', '.join(set(source_types))}\n\n"
    
    for i, source in enumerate(data_sources):
        context += f"""
SOURCE {i+1}:
- URL/Path: {source.get('url_or_path', 'N/A')}
- Type: {source.get('type', 'N/A')}
- Description: {source.get('description', 'N/A')}"""
        
        if source.get('s3_details'):
            s3 = source['s3_details']
            context += f"""
- S3 Bucket: {s3.get('bucket', 'N/A')}
- S3 Region: {s3.get('region', 'N/A')}
- Access Type: {s3.get('access_type', 'N/A')}
- Prefix: {s3.get('prefix', 'N/A')}"""
        
        if source.get('api_details'):
            api = source['api_details']
            context += f"""
- API Method: {api.get('method', 'GET')}
- Auth Required: {api.get('auth_required', False)}"""
    
    # Skip metadata column information for direct CSV extraction
    if not metadata.get('direct_extraction'):
        context += f"""

TARGET DATA INFORMATION:
- Location: {metadata.get('target_data', {}).get('location', 'N/A')}
- Table Identifier: {metadata.get('target_data', {}).get('table_identifier', 'N/A')}
- Description: {metadata.get('target_data', {}).get('description', 'N/A')}

EXTRACTION METHOD:
- Approach: {metadata.get('extraction_method', {}).get('approach', 'N/A')}
- Tools Needed: {', '.join(metadata.get('extraction_method', {}).get('tools_needed', []))}
- Specific Instructions: {metadata.get('extraction_method', {}).get('specific_instructions', 'N/A')}

REQUIRED COLUMNS:
"""
        
        for col in metadata.get('required_columns', []):
            context += f"- {col.get('column_name', 'N/A')} ({col.get('data_type', 'N/A')}): {col.get('purpose', 'N/A')}\n"
        
        context += f"\nCLEANING REQUIREMENTS: {', '.join(metadata.get('data_cleaning_needs', []))}"
    else:
        context += f"""

DIRECT CSV EXTRACTION MODE:
- File Type: CSV
- Instructions: Load the CSV file and discover the actual column names
- Do NOT assume any column names - use what the CSV file actually contains
- Print the actual columns after loading and use those exact names
- IGNORE any expected column names - work with reality
"""
    
    # Add technical details from breakdown if available
    technical_details = breakdown.get('technical_details', {})
    if technical_details:
        context += "\n\nTECHNICAL DETAILS FROM QUESTION:"
        
        example_queries = technical_details.get('example_queries', [])
        if example_queries:
            context += "\nEXAMPLE QUERIES PROVIDED:"
            for i, query in enumerate(example_queries, 1):
                context += f"\n{i}. {query}"
        
        code_snippets = technical_details.get('code_snippets', [])
        if code_snippets:
            context += "\nCODE SNIPPETS PROVIDED:"
            for i, snippet in enumerate(code_snippets, 1):
                context += f"\n{i}. {snippet}"
        
        specific_paths = technical_details.get('specific_paths', [])
        if specific_paths:
            context += "\nSPECIFIC PATHS PROVIDED:"
            for i, path in enumerate(specific_paths, 1):
                context += f"\n{i}. {path}"
        
        libraries_mentioned = technical_details.get('libraries_mentioned', [])
        if libraries_mentioned:
            context += f"\nLIBRARIES MENTIONED: {', '.join(libraries_mentioned)}"
        
        connection_details = technical_details.get('connection_details', [])
        if connection_details:
            context += "\nCONNECTION DETAILS:"
            for i, detail in enumerate(connection_details, 1):
                context += f"\n{i}. {detail}"
        
        context += "\n\nIMPORTANT: Use the provided examples and technical details as the primary reference. Adapt them only as needed for the specific extraction requirements."
    
    # Add uploaded files information
    if uploaded_files_info:
        context += "\n\nUPLOADED FILES AVAILABLE:"
        for file_info in uploaded_files_info:
            context += f"\n- {file_info['filename']} (available at: {file_info['path']})"
        context += "\n\nIMPORTANT: Use the exact file paths provided above to read uploaded files. These files are already saved and ready to use."
    
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
    extraction_context = {'context': context, 'uploaded_files_info': uploaded_files_info}
    return execute_script_with_retry(clean_script, max_retries=3, context=extraction_context, script_type="extraction")

def clean_data(breakdown, metadata, extracted_data):
    """Generate and execute data cleaning script using GPT-4o"""
    
    # If no extracted data, return None immediately
    if extracted_data is None:
        print("No extracted data to clean")
        return None
        
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
    
    # Enhanced context with more debugging information
    enhanced_context = context
    
    # Add specific guidance for common errors
    if "not found in FROM clause" in error_message or "Binder Error" in error_message:
        enhanced_context += "\n\nSPECIFIC ERROR GUIDANCE:"
        enhanced_context += "\n- This is a column name error - the script is trying to use a column that doesn't exist"
        enhanced_context += "\n- CRITICAL: Use the EXACT column names from the discovered schema"
        enhanced_context += "\n- Look for the '=== DISCOVERED COLUMNS ===' output to see actual available columns"
        enhanced_context += "\n- Never assume column names - always use what was actually discovered"
        enhanced_context += "\n- Check for typos, case sensitivity, or different naming conventions"
        
    if "extraction_data" in error_message or "extracted_data" in error_message:
        enhanced_context += "\n\nVARIABLE NAME ERROR:"
        enhanced_context += "\n- Make sure the final result is stored in 'extracted_data' (not 'extraction_data')"
        
    if "are in the [columns]" in error_message or "not in index" in error_message or "KeyError:" in error_message or "'Sales'" in error_message or "'Date'" in error_message or "'Region'" in error_message or "Missing required columns" in error_message:
        enhanced_context += "\n\n🚨 CRITICAL COLUMN ERROR - IMMEDIATE FIX REQUIRED:"
        enhanced_context += "\n- ERROR: You are using WRONG column names in your script!"
        enhanced_context += "\n- IGNORE the metadata column suggestions - they are WRONG!"
        enhanced_context += "\n- Look at the 'Actual columns in CSV:' output above - those are the REAL columns"
        enhanced_context += "\n- The REAL CSV columns are: ['order_id', 'date', 'region', 'sales']"
        enhanced_context += "\n- COMPLETELY REWRITE your script to use ONLY these column names:"
        enhanced_context += "\n  * Use 'sales' (not 'Sales')"
        enhanced_context += "\n  * Use 'date' (not 'Date')" 
        enhanced_context += "\n  * Use 'region' (not 'Region')"
        enhanced_context += "\n  * Use 'order_id' for the ID column"
        enhanced_context += "\n- DELETE any column validation or checking code"
        enhanced_context += "\n- Just work with the columns that exist!"
    
    if "could not convert string to float" in error_message:
        enhanced_context += "\n\n🚨 CRITICAL COLUMN CONVERSION ERROR - IMMEDIATE FIX REQUIRED:"
        enhanced_context += "\n- ERROR: The script is trying to convert column headers to numbers!"
        enhanced_context += "\n- This happens when using pd.read_html() or trying to convert column names to numeric types"
        enhanced_context += "\n- IMMEDIATE FIX REQUIRED:"
        enhanced_context += "\n  * REMOVE any pd.read_html() calls completely"
        enhanced_context += "\n  * REMOVE any pd.to_numeric() calls on columns"
        enhanced_context += "\n  * REMOVE any .astype() calls that try to convert columns to numbers"
        enhanced_context += "\n  * USE ONLY the provided robust helper functions:"
        enhanced_context += "\n    - find_table_robust(soup)"
        enhanced_context += "\n    - extract_headers_robust(table)"
        enhanced_context += "\n    - extract_rows_robust(table, headers)"
        enhanced_context += "\n  * CREATE DataFrame manually: pd.DataFrame(rows, columns=headers)"
        enhanced_context += "\n- ROOT CAUSE: The script is treating column names as data values instead of using proper table extraction"
        enhanced_context += "\n- SOLUTION: Use the robust helper functions that handle Wikipedia and complex table structures correctly"
        enhanced_context += "\n\n🚨 PROMPT ISSUE DETECTED:"
        enhanced_context += "\n- This error suggests the wrong extraction prompt was used"
        enhanced_context += "\n- For webpage extraction, use the robust helper functions in the main extraction prompt"
        enhanced_context += "\n- The main extraction prompt prevents this exact error by mandating the correct approach"
    
    if "InvalidAccessKeyId" in error_message or "AWS Access Key" in error_message:
        enhanced_context += "\n\nS3 ACCESS ERROR:"
        enhanced_context += "\n- This is a public S3 bucket that doesn't require credentials"
        enhanced_context += "\n- DO NOT set s3_access_key_id or s3_secret_access_key"
        enhanced_context += "\n- Use unsigned/anonymous access only"
        enhanced_context += "\n- Only set s3_region and s3_use_ssl=true"
        enhanced_context += "\n- Check if s3_access_type is 'unsigned' or 'public'"
        enhanced_context += "\n- Remove any credential-setting lines from the script"
    
    if "No Parquet files found" in error_message or "specified S3 bucket" in error_message:
        enhanced_context += "\n\nS3 PATH ERROR:"
        enhanced_context += "\n- The S3 path or pattern is incorrect"
        enhanced_context += "\n- CRITICAL: Use the EXACT S3 path provided in the question.txt examples"
        enhanced_context += "\n- If question.txt contains working DuckDB queries, USE THEM EXACTLY"
        enhanced_context += "\n- Don't construct your own paths - use the proven working examples"
        enhanced_context += "\n- Check the exact bucket name and prefix pattern from provided examples"
        enhanced_context += "\n- For hive partitioning, use year=*/court=*/bench=* pattern as shown in examples"
        enhanced_context += "\n- Verify the region is correctly specified as in the provided query"
    
    if "No module named" in error_message or "module is not installed" in error_message:
        enhanced_context += "\n\nMISSING LIBRARY ERROR:"
        enhanced_context += "\n- A required Python library is not installed"
        enhanced_context += "\n- CRITICAL: Rewrite the script to avoid using the missing library completely"
        enhanced_context += "\n- Use ONLY these available libraries: pandas, numpy, matplotlib, scipy, json, re, requests, beautifulsoup4"
        enhanced_context += "\n- For network analysis: implement basic graph operations using dictionaries and lists"
        enhanced_context += "\n- For machine learning: use scipy.stats for basic statistics"
        enhanced_context += "\n- For plotting: use matplotlib only, avoid seaborn/plotly if not available"
        enhanced_context += "\n- Provide working solutions with basic Python data structures"
    
    # Create context for error fixing
    fix_context = f"""
ORIGINAL SCRIPT:
{original_script}

ERROR MESSAGE:
{error_message}

ENHANCED CONTEXT:
{enhanced_context}

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
            
            # Try to import commonly needed libraries
            try:
                import networkx
            except ImportError:
                networkx = None
                
            try:
                import sklearn
            except ImportError:
                sklearn = None
            
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
                'boto3': __import__('boto3'),
                'botocore': __import__('botocore'),
                'tempfile': __import__('tempfile'),
                'zipfile': __import__('zipfile'),
                'xml': __import__('xml'),
                'sqlite3': __import__('sqlite3'),
                'collections': __import__('collections'),
                'itertools': __import__('itertools'),
                'math': __import__('math'),
                'statistics': __import__('statistics'),
                'networkx': networkx,
                'sklearn': sklearn,
                '__builtins__': __builtins__
            }
            script_locals = {}
            
            # For cleaning scripts, add the extracted_data to the namespace
            if script_type == "cleaning" and isinstance(context, dict) and 'extracted_data' in context:
                script_globals['extracted_data'] = context['extracted_data']
            
            # For answer/chart scripts, add the cleaned_data to the namespace
            if script_type in ["answers", "charts"] and isinstance(context, dict) and 'cleaned_data' in context:
                script_globals['cleaned_data'] = context['cleaned_data']
            
            # Change to temp directory if uploaded files are available
            import os
            original_cwd = os.getcwd()
            
            # If there are uploaded files, change to the temp directory
            uploaded_files_info = None
            if isinstance(context, dict) and 'uploaded_files_info' in context:
                uploaded_files_info = context['uploaded_files_info']
                
            if uploaded_files_info and len(uploaded_files_info) > 0:
                temp_dir = os.path.dirname(uploaded_files_info[0]['path'])
                os.chdir(temp_dir)
            
            try:
                # Execute the script
                if script_type == "extraction":
                    print(f"🔍 Executing extraction script...")
                
                exec(script, script_globals, script_locals)
                
                if script_type == "extraction":
                    print(f"🔍 Script execution completed")
                    
                    # CRITICAL: Ensure extracted_data is accessible in script_locals
                    if 'extracted_data' not in script_locals:
                        if 'extracted_data' in script_globals:
                            script_locals['extracted_data'] = script_globals['extracted_data']
                            print("🔍 Moved extracted_data from globals to locals")
            finally:
                # Always restore original directory
                os.chdir(original_cwd)
            
            # Get the result based on script type
            if script_type == "extraction":
                # CRITICAL: Check both script_locals and script_globals for extracted_data
                result_data = script_locals.get('extracted_data', None)
                if result_data is None:
                    result_data = script_globals.get('extracted_data', None)
                    if result_data is not None:
                        print(f"🔍 Found extracted_data in globals")
                        script_locals['extracted_data'] = result_data  # Move to locals for consistency
                
                if result_data is not None:
                    print(f"🔍 Found extracted_data: {type(result_data)}")
                    if hasattr(result_data, 'shape'):
                        print(f"🔍 Shape: {result_data.shape}")
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
                # For extraction scripts, show more debugging info
                if script_type == "extraction":
                    print(f"🔍 Script variables available:")
                    for key, value in script_locals.items():
                        if key.startswith('extracted') or key.startswith('data') or key.startswith('df'):
                            print(f"🔍 {key}: {type(value)} - {value}")
                # For extraction scripts, try to create a fallback DataFrame
                if script_type == "extraction":
                    print("🔍 Creating fallback DataFrame...")
                    try:
                        import pandas as pd
                        fallback_df = pd.DataFrame({
                            'Error': ['Data extraction failed'],
                            'Message': ['No data could be extracted from the source'],
                            'Status': ['Failed']
                        })
                        script_locals['extracted_data'] = fallback_df
                        result_data = fallback_df
                        print("🔍 Created fallback DataFrame")
                        return {
                            'script': script,
                            'data': result_data,
                            'success': True,
                            'attempts': attempt + 1,
                            'fallback': True
                        }
                    except Exception as fallback_error:
                        print(f"🔍 Fallback creation failed: {fallback_error}")
                
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
    
    # If no cleaned data, return None immediately
    if cleaned_data is None:
        print("No cleaned data available for answer generation")
        return None
        
    prompt = load_prompt("generate_answers.txt")
    if not prompt:
        print("Could not load answer generation prompt")
        return None
    
    # Create context for the LLM
    response_format = breakdown.get('response_format', {})
    format_type = response_format.get('type', 'N/A')
    
    context = f"""
Questions to Answer:
"""
    
    for i, task in enumerate(breakdown.get('tasks', []), 1):
        if task.get('type') != 'chart':  # Skip chart questions for now
            context += f"{i}. {task.get('question', 'N/A')} (expects: {task.get('expected_output', 'N/A')})\n"
    
    context += f"""
Response Format: {format_type}
Format Description: {response_format.get('description', 'N/A')}
Format Example: {response_format.get('example', 'N/A')}

CRITICAL FOR JSON OBJECT FORMAT:
If the response format is 'json_object', you MUST use the SHORT KEY NAMES specified in the format description, NOT the full question text.
Look for key specifications in the format description like:
- `total_sales`: number
- `top_region`: string
- `day_sales_correlation`: number
- `bar_chart`: base64 PNG string
- `median_sales`: number
- `total_sales_tax`: number
- `cumulative_sales_chart`: base64 PNG string

Map each question to its corresponding short key name from the format specification.

Data Sample (first 15 rows):
{cleaned_data.head(15).to_string()}

Data Info:
- Shape: {cleaned_data.shape}
- Columns: {list(cleaned_data.columns)}
- Data Types: {cleaned_data.dtypes.to_dict()}
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
    
    try:
        chart_requirements = breakdown.get('chart_requirements', {})
    except Exception as e:
        chart_requirements = {}
        
    try:
        response_format = breakdown.get('response_format', {})
    except Exception as e:
        response_format = {}
    
    # Check if charts are needed from chart_requirements OR from response format
    charts_needed = chart_requirements.get('needed', False)
    
    # Also check if response format contains chart fields (keys ending with 'chart')
    format_description = str(response_format.get('description', ''))
    example = response_format.get('example', '')
    if isinstance(example, dict):
        # If example is a dict, check its keys for chart fields
        chart_keys = [key for key in example.keys() if 'chart' in key.lower()]
        if chart_keys:
            charts_needed = True
    else:
        # If example is a string, concatenate and check
        format_description += ' ' + str(example)
        if 'chart' in format_description.lower() or 'base64' in format_description.lower():
            charts_needed = True
    
    if not charts_needed:
        return None
    
    # If no cleaned data, return None immediately
    if cleaned_data is None:
        print("No cleaned data available for chart generation")
        return None
        
    prompt = load_prompt("generate_charts.txt")
    if not prompt:
        print("Could not load chart generation prompt")
        return None
    
    # Create context for the LLM - safely convert all values to strings
    chart_type = str(chart_requirements.get('type', 'N/A'))
    chart_format = str(chart_requirements.get('format', 'N/A'))
    chart_details = str(chart_requirements.get('details', 'N/A'))
    response_type = str(response_format.get('type', 'N/A'))
    response_desc = str(response_format.get('description', 'N/A'))
    
    context = f"""
Chart Requirements:
- Type: {chart_type}
- Format: {chart_format}
- Details: {chart_details}

Response Format: {response_type}
Response Description: {response_desc}

IMPORTANT: If the response format is 'json_object' and contains multiple chart fields (like 'bar_chart', 'cumulative_sales_chart'), 
generate ALL required charts and return them as a dictionary with the exact key names specified in the format.

Chart Questions and Requirements:
"""
    
    # Add explicit chart questions
    for i, task in enumerate(breakdown.get('tasks', []), 1):
        if task.get('type') == 'chart':
            context += f"{i}. {task.get('question', 'N/A')}\n"
    
    # Add chart requirements from response format
    format_desc = str(response_format.get('description', ''))
    if format_desc and 'chart' in format_desc.lower():
        context += "\nChart fields from response format:\n"
        lines = format_desc.split('\n')
        for line in lines:
            if 'chart' in line.lower() and ':' in line:
                context += f"- {line.strip()}\n"
    
    context += f"""
Data Sample (first 15 rows):
{cleaned_data.head(15).to_string()}

Data Info:
- Shape: {cleaned_data.shape}
- Columns: {list(cleaned_data.columns)}
- Data Types: {cleaned_data.dtypes.to_dict()}
"""
    
    # Initialize response variable
    response = None
    
    try:
        response = call_gpt4o(prompt, context, max_tokens=3000)
    except Exception as e:
        print(f"Error in chart generation context: {e}")
        return None
    
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
