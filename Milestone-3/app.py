# Important libraries

import subprocess  # For running external processes
import os  # For interacting with the operating system
import pandas as pd  # For working with tabular data
import gradio as gr  # For creating interactive interfaces
import matplotlib.pyplot as plt  # For plotting data visualizations
import re  # For regular expression operations
from dotenv import load_dotenv  # For loading environment variables

# Load environment variables from a .env file
load_dotenv()
api_key = os.getenv('apiKey')  # Retrieve the API key from environment variables

# Enhanced function to extract numeric values from text
def extract_numeric_values(text):
    """
    Removes currency symbols and commas from text, extracts numeric values,
    and converts them to float if possible.

    Args:
        text (str): The input text containing numeric data.

    Returns:
        list: A list of numeric values (float) extracted from the text.
    """
    text = re.sub(r'[₹$,]', '', text)  # Remove currency symbols and commas
    numbers = re.findall(r'\d+\.?\d*', text)  # Find all numeric patterns

    try:
        return [float(num) for num in numbers]  # Convert to float
    except:
        return []  # Return an empty list if conversion fails

# Sanitize text by removing unwanted symbols
def sanitize_text(text):
    """
    Cleans input text by removing special characters and extra whitespace.

    Args:
        text (str): The text to sanitize.

    Returns:
        str: Sanitized text.
    """
    return re.sub(r'[\*\u201A\u20B9]', '', str(text)).strip()

# Function to perform OCR using llama-ocr with prompts
def run_ocr_with_prompt(file_path, prompt):
    """
    Runs the OCR process on a given file with a specific prompt using the llama-ocr tool.

    Args:
        file_path (str): Path to the document to be processed.
        prompt (str): The prompt that specifies the type of data to extract.

    Returns:
        str: The raw output from the OCR process.
    """
    try:
        result = subprocess.run(
            ['node', 'ocrScript.js', file_path, api_key, prompt],
            capture_output=True,
            text=True,
            timeout=60  # Set a timeout for the OCR process
        )
        output = result.stdout.strip()  # Capture and clean the output
        print(f"Raw OCR Output: {output}")
        return output
    except subprocess.TimeoutExpired:
        return "Error: OCR process timed out"  # Handle timeout errors
    except Exception as e:
        return f"Error: {str(e)}"  # Handle other exceptions

# Dynamic field extraction based on document type
def extract_fields(document_type, file_path):
    """
    Extracts specific fields from a document based on its type using predefined prompts.

    Args:
        document_type (str): The type of document (e.g., Salary Slip, Invoice).
        file_path (str): Path to the document file.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted fields and their values.
    """
    # Predefined prompts for different document types
    prompts = {
        "Salary Slip": """Carefully extract the following details from the salary slip:
        Locate and extract ONLY these values: Basic Salary, HRA, Conveyance, Special Allowance, Net Salary""",
        "Bank Statement": """Extract precise financial values: Withdrawals , Deposits, Balance""",
        "Balance Sheet": """Extract exact numeric values: Assets, Liabilities, Equity, Working capital""",
        "Profit and Loss": """Extract critical financial metrics: Revenue, Expenses, Net Profit, Gross Profit""",
    }

    prompt_response = run_ocr_with_prompt(file_path, prompts.get(document_type, ""))  # Get OCR response
    extracted_data = {}

    try:
        # Parse OCR response line by line
        for line in prompt_response.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                numeric_values = extract_numeric_values(value)  # Extract numeric values
                extracted_data[key] = numeric_values[0] if numeric_values else value  # Use first numeric value

        # Convert the extracted data into a DataFrame
        df = pd.DataFrame.from_dict(extracted_data, orient='index', columns=['Value']).reset_index()
        df.columns = ['Field', 'Value']
        return df
    except Exception as e:
        print(f"Error parsing extraction results: {e}")
        return pd.DataFrame(columns=['Field', 'Value'])  # Return empty DataFrame on error

# Function to generate visualizations
def generate_visualizations(df, document_type):
    """
    Generates pie and bar charts to visualize the extracted data.

    Args:
        df (pd.DataFrame): The DataFrame containing extracted data.
        document_type (str): The type of document.

    Returns:
        str: The file path of the generated visualization image, or None if generation fails.
    """
    plt.close('all')  # Close any existing plots
    if df.empty or len(df) < 2:  # Check if data is sufficient for visualization
        return None

    try:
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce').dropna()  # Convert values to numeric
        df['Field'] = df['Field'].str.replace(r'[\*]', '', regex=True)  # Clean field labels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Create subplots

        # Pie Chart
        df.plot(kind='pie', y='Value', labels=df['Field'], ax=ax1, autopct='%1.1f%%', startangle=90, legend=False)
        ax1.set_title(f'{document_type} Distribution')

        # Bar Chart
        df.plot(kind='bar', x='Field', y='Value', ax=ax2, color='skyblue', rot=45, legend=False)
        ax2.set_title('Numeric Values')
        ax2.set_xlabel('Fields')
        ax2.set_ylabel('Amount')

        plt.tight_layout()
        chart_path = "visualization_charts.png"
        plt.savefig(chart_path)
        plt.close()
        return chart_path
    except Exception as e:
        print(f"Visualization error: {e}")
        return None

# Function to process OCR results and extract data in tabular format
def extract_data(document_type, files):
    """
    Processes uploaded files, extracts raw OCR text and specific fields,
    and combines the results into DataFrames.

    Args:
        document_type (str): The type of document.
        files (list): List of uploaded file objects.

    Returns:
        tuple: A tuple containing the raw data DataFrame and the extracted fields DataFrame.
    """
    all_raw_data = []
    all_extracted_data = []

    for file in files:
        raw_text = run_ocr_with_prompt(file.name, "Extract the full text from the document.")  # Get raw text
        for line in raw_text.split("\n"):
            sanitized_line = sanitize_text(line.strip())
            if sanitized_line:
                all_raw_data.append({"Line": sanitized_line})

        extracted_fields_df = extract_fields(document_type, file.name)  # Extract specific fields
        all_extracted_data.append(extracted_fields_df)

    raw_data_df = pd.DataFrame(all_raw_data)
    extracted_data_df = pd.concat(all_extracted_data, ignore_index=True) if all_extracted_data else pd.DataFrame()
    return raw_data_df, extracted_data_df

# Gradio interface function
def interface(document_type, files):
    """
    Interface function for Gradio to process files and return data and visualizations.

    Args:
        document_type (str): The type of document.
        files (list): List of uploaded files.

    Returns:
        tuple: Raw data DataFrame, extracted data DataFrame, and chart path.
    """
    raw_data_df, extracted_data_df = extract_data(document_type, files)
    chart_path = generate_visualizations(extracted_data_df, document_type) if not extracted_data_df.empty else None
    return raw_data_df, extracted_data_df, chart_path

# Define Gradio interface
document_types = ["Salary Slip", "Bank Statement", "Balance Sheet", "Invoice", "Profit and Loss"]
interface = gr.Interface(
    fn=interface,
    inputs=[
        gr.Dropdown(choices=document_types, label="Document Type"),
        gr.File(label="Upload Documents", file_types=["image"], file_count="multiple")
    ],
    outputs=[
        gr.Dataframe(label="Full Extracted Text (Raw Data)"),
        gr.Dataframe(label="Specific Extracted Data"),
        gr.Image(label="Charts (Pie & Bar)"),
    ],
    title="Llama OCR with Data Visualization"
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
