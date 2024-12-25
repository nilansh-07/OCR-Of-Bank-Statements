import subprocess  # For running external processes
import os  # For interacting with the operating system
import pandas as pd  # For working with tabular data
import gradio as gr  # For creating interactive interfaces
import matplotlib.pyplot as plt  # For plotting data visualizations
import re  # For regular expression operations
from dotenv import load_dotenv  # For loading environment variables
import cloudinary  # For interacting with the Cloudinary API
from cloudinary.api import resources  # For accessing Cloudinary resources

# Load environment variables from a .env file
load_dotenv()
api_key = os.getenv('apiKey')  # Retrieve the API key of llama-ocr from environment variables

# Configure Cloudinary using credentials from environment variables
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),  # Cloudinary cloud name
    api_key=os.getenv('CLOUDINARY_API_KEY'),  # Cloudinary API key
    api_secret=os.getenv('CLOUDINARY_API_SECRET')  # Cloudinary API secret
)

# Define folder options for organizing documents on Cloudinary
FOLDERS = [
    "BalanceSheets",
    "BankStatements",
    "Invoices",
    "ProfitandLossStatements",
    "SalarySlips",
]

# Fetch images from a specified Cloudinary folder
def fetch_images(folder, limit):
    try:
        # Search for images in the specified folder
        result = cloudinary.Search().expression(f"folder:{folder}").max_results(limit).execute()
        image_urls = [resource['secure_url'] for resource in result.get('resources', [])]  # Extract image URLs

        if not image_urls:  # Handle case where no images are found
            return "No images found for the selected folder.", []

        return f"Fetched {len(image_urls)} images from folder '{folder}'", image_urls
    except Exception as e:
        return f"Error fetching images: {str(e)}", []

# Extract numeric values from text (e.g., monetary or other numeric data)
def extract_numeric_values(text):
    text = re.sub(r'[\u20B9$,]', '', text)  # Remove currency symbols and commas
    numbers = re.findall(r'\d+\.?\d*', text)  # Find all numeric patterns

    try:
        return [float(num) for num in numbers]  # Convert to float
    except:
        return []  # Return empty list if conversion fails

# Sanitize text by removing unwanted symbols
def sanitize_text(text):
    return re.sub(r'[\*\u201A\u20B9]', '', str(text)).strip()

# Perform OCR using llama-ocr with prompts
def run_ocr_with_prompt(file_path, prompt):
    try:
        # Run external OCR script
        result = subprocess.run(
            ['node', 'ocrScript.js', file_path, api_key, prompt],
            capture_output=True,
            text=True,
            timeout=120  # Timeout after 2 minutes
        )
        output = result.stdout.strip()  # Retrieve OCR output
        print(f"Raw OCR Output: {output}")
        return output
    except subprocess.TimeoutExpired:
        return "Error: OCR process timed out"  # Handle timeout error
    except Exception as e:
        return f"Error: {str(e)}"  # Handle other errors

# Extract relevant fields from documents based on their type
def extract_fields(document_type, file_path):
    # Define prompts for various document types
    prompts = {
        "Salary Slip": "Extract the following details from the salary slip: Basic Salary, House Rent Allowance (HRA), Provident Fund (PF) Deduction, and Net Pay.",
        "Bank Statement": "Extract the following details from the bank statement: Account Balance, Deposits, Withdrawals, and Transaction Amounts.",
        "Balance Sheet": "Extract the following details from the balance sheet: Total Assets, Total Liabilities, Equity, and Retained Earnings.",
        "Invoice": "Extract the following details from the invoice: Subtotal, Tax Amount, Discounts, and Total Invoice Amount.",
        "Profit and Loss": "Extract the following details from the profit and loss statement: Revenue, Cost of Goods Sold (COGS), Gross Profit, and Net Profit."
    }

    # Perform OCR with the appropriate prompt
    prompt_response = run_ocr_with_prompt(file_path, prompts.get(document_type, ""))
    extracted_data = {}

    try:
        # Parse OCR output and extract fields
        for line in prompt_response.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)  # Split key-value pairs
                key = key.strip()
                value = value.strip()
                numeric_values = extract_numeric_values(value)  # Extract numeric values
                extracted_data[key] = numeric_values[0] if numeric_values else value

        # Convert extracted data into a DataFrame
        df = pd.DataFrame.from_dict(extracted_data, orient='index', columns=['Value']).reset_index()
        df.columns = ['Field', 'Value']
        return df
    except Exception as e:
        print(f"Error parsing extraction results: {e}")
        return pd.DataFrame(columns=['Field', 'Value'])

# Generate visualizations (pie and bar charts) for extracted data
def generate_visualizations(df, document_type):
    if df.empty or len(df) < 2:  # Check if DataFrame has enough data
        return None

    try:
        # Prepare data for plotting
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['Value'])  # Drop invalid entries
        df['Field'] = df['Field'].str.replace(r'[\*]', '', regex=True)

        # Create visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f'{document_type} Visualizations', fontsize=16, fontweight='bold')

        # Pie chart
        df.plot(kind='pie', y='Value', labels=df['Field'], ax=ax1, autopct='%1.1f%%', startangle=90, legend=False, colors=plt.cm.Paired.colors)
        ax1.set_title('Value Distribution (Pie)', fontsize=14)

        # Bar chart
        df.plot(kind='bar', x='Field', y='Value', ax=ax2, color='skyblue', edgecolor='black')
        ax2.set_title('Field-wise Values (Bar)', fontsize=14)

        chart_path = "visualization_charts.png"
        plt.savefig(chart_path, dpi=300)  # Save chart to file
        plt.close()
        return chart_path
    except Exception as e:
        print(f"Visualization error: {e}")
        return None

# Process OCR for multiple images and extract data
def process_ocr_for_images(document_type, image_urls):
    all_raw_data = []
    all_extracted_data = []

    for url in image_urls:
        file_path = "visual_image.png"  # Temporary file path for images
        os.system(f"curl -o {file_path} {url}")  # Download image

        raw_text = run_ocr_with_prompt(file_path, "Extract the full text from the document.")  # Perform OCR
        for line in raw_text.split("\n"):
            sanitized_line = sanitize_text(line.strip())
            if sanitized_line:
                all_raw_data.append({"Extracted Data": sanitized_line})  # Collect raw text

        extracted_fields_df = extract_fields(document_type, file_path)  # Extract fields
        all_extracted_data.append(extracted_fields_df)

    raw_data_df = pd.DataFrame(all_raw_data)  # Raw OCR output
    extracted_data_df = pd.concat(all_extracted_data, ignore_index=True) if all_extracted_data else pd.DataFrame()  # Extracted fields
    return raw_data_df, extracted_data_df

# Gradio interface for fetching and processing data
def interface(folder, limit, document_type):
    message, image_urls = fetch_images(folder, limit)
    if not image_urls:
        return pd.DataFrame(), pd.DataFrame(), None, message, []

    raw_data_df, extracted_data_df = process_ocr_for_images(document_type, image_urls)  # Process OCR
    chart_path = generate_visualizations(extracted_data_df, document_type) if not extracted_data_df.empty else None
    return raw_data_df, extracted_data_df, chart_path, message, image_urls

# Create Gradio UI
with gr.Blocks() as ui:
    gr.Markdown("# Fetches Images from Cloudinary with OCR Integration and Visualization")
    folder_input = gr.Dropdown(choices=FOLDERS, label="Select Folder")
    limit_input = gr.Number(label="Number of Images", value=5, precision=0)
    document_type_input = gr.Dropdown(choices=["Balance Sheet", "Bank Statement", "Invoice", "Profit and Loss", "Salary Slip"], label="Document Type")

    fetch_button = gr.Button("Fetch and Process Images")
    output_message = gr.Textbox(label="Output Message", interactive=False)
    output_images = gr.Gallery(label="Fetched Images")
    output_raw_data = gr.Dataframe(label="Full Extracted Text (Raw Data)")
    output_extracted_data = gr.Dataframe(label="Specific Extracted Data")
    output_chart = gr.Image(label="Charts (Pie & Bar)")

    # Bind interface inputs and outputs
    fetch_button.click(
        interface, 
        inputs=[folder_input, limit_input, document_type_input], 
        outputs=[output_raw_data, output_extracted_data, output_chart, output_message, output_images]
    )

# Launch Gradio app
ui.launch()
