# Import libraries
import pdfplumber
import spacy
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from google.colab import drive
import time

# Install missing libraries quietly
print("Installing pdfplumber and seaborn...")
!pip install -q pdfplumber seaborn

# Load spaCy model with error handling
print("Downloading spaCy model...")
try:
    !python -m spacy download en_core_web_sm -q
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully")
except Exception as e:
    print(f"Failed to load spaCy model: {e}")
    nlp = None

# Setup T5 model with fallback
print("Loading T5 model...")
try:
    model_name = "t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    simplifier = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    print("T5 model loaded successfully")
except Exception as e:
    print(f"Failed to load T5 model: {e}")
    simplifier = None

# Step 1: Mount Google Drive and Extract Text from PDF
def extract_text(pdf_path="/content/drive/MyDrive/Report2.pdf"):
    """Extracts text from a PDF in Google Drive with robust error handling."""
    print("Mounting Google Drive...")
    try:
        start_time = time.time()
        drive.mount('/content/drive', force_remount=True)
        print(f"Drive mounted in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Drive mount failed: {e}")
        return ""

    print(f"Opening {pdf_path}...")
    try:
        start_time = time.time()
        with pdfplumber.open(pdf_path) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages[:5])  # Limit to 5 pages
        print(f"Text extracted in {time.time() - start_time:.2f} seconds")
        return text
    except FileNotFoundError:
        print(f"File not found at {pdf_path}. Check path and permissions.")
        return ""
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""

# Step 2: Extract Financial Metrics and Entities (Rupees Only, Fixed)
def extract_financial_data(text):
    """Extracts financial data for INR only with corrected regex."""
    if not nlp:
        print("NLP unavailable—skipping entity extraction")
        return defaultdict(list), {}

    print("Extracting financial data (INR only)...")
    start_time = time.time()
    doc = nlp(text[:5000])  # Limit to 5k chars
    entities = defaultdict(list)
    metrics = {}

    for ent in doc.ents:
        if ent.label_ in ["ORG", "MONEY", "DATE"]:
            entities[ent.label_].append(ent.text)

    # Corrected INR patterns (single capture group for number)
    inr_patterns = {
        "Revenue": r"revenue.*?[₹]?[Rr][Ss]\.?|INR\s*([\d,.]+)\s*(?:lakh|crore|rupees|INR)?",
        "Expenses": r"(expense|cost).*?[₹]?[Rr][Ss]\.?|INR\s*([\d,.]+)\s*(?:lakh|crore|rupees|INR)?",
        "Profit": r"(profit|net income).*?[₹]?[Rr][Ss]\.?|INR\s*([\d,.]+)\s*(?:lakh|crore|rupees|INR)?",
        "Debt": r"(debt|liabilit).*?[₹]?[Rr][Ss]\.?|INR\s*([\d,.]+)\s*(?:lakh|crore|rupees|INR)?"
    }

    # Conversion multipliers
    multipliers = {
        "lakh": 1e5,
        "crore": 1e7,
        "rupees": 1,
        "inr": 1  # Base unit
    }

    # Process INR patterns
    for key, pattern in inr_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            cleaned_values = []
            for match in matches:
                # Since we have one capture group, match is a string or empty
                if isinstance(match, str) and match:  # Ensure match is non-empty
                    num_str = match.replace(",", "").rstrip(".")
                    try:
                        value = float(num_str)
                        # Check context for multipliers
                        context_match = re.search(pattern + r".*?(lakh|crore|rupees|INR)?", text, re.IGNORECASE)
                        if context_match:
                            context = context_match.group(0).lower()
                            for mult_name, mult_value in multipliers.items():
                                if mult_name in context:
                                    value *= mult_value
                                    break
                        cleaned_values.append(value)
                    except ValueError:
                        print(f"Skipping invalid INR number: '{num_str}'")
                        continue
            if cleaned_values:
                metrics[key] = cleaned_values

    print(f"Data extracted in {time.time() - start_time:.2f} seconds")
    return entities, metrics

# Step 3: Simplify Text and Generate Layman’s Story (Rupees Only)
def simplify_and_story(text, entities, metrics):
    """Simplifies text and generates a story for INR only."""
    if not simplifier:
        print("Simplifier unavailable—using raw text")
        return text[:100], "Story generation skipped due to model error."

    print("Simplifying text and generating story...")
    start_time = time.time()
    chunk = text[:250]
    try:
        simplified = simplifier(f"summarize: {chunk}", max_length=50, min_length=10, do_sample=False)[0]["generated_text"]
    except Exception as e:
        print(f"Simplification failed: {e}")
        simplified = chunk

    story = ["Here’s the story of this company in simple terms:"]
    if "ORG" in entities:
        story.append(f"This company, {entities['ORG'][0] if entities['ORG'] else 'unnamed'}, has been making moves.")

    if "Revenue" in metrics:
        rev = metrics["Revenue"][0]
        story.append(f"They earned ₹{rev:,.0f} in sales—a hefty sum!")
    if "Expenses" in metrics:
        exp = metrics["Expenses"][0]
        story.append(f"But they spent ₹{exp:,.0f}, eating into their funds.")
    if "Profit" in metrics:
        prof = metrics["Profit"][0]
        story.append(f"They ended up with ₹{prof:,.0f} as profit—not bad!")
    if "Debt" in metrics:
        debt = metrics["Debt"][0]
        story.append(f"Careful, though—they owe ₹{debt:,.0f}, a big burden.")

    story.append("Compared to what they say in public, this might not be the full picture—dig deeper!")
    print(f"Story generated in {time.time() - start_time:.2f} seconds")
    return simplified, "\n".join(story)

# Step 4: Technical Write-Up (Rupees Only)
def technical_writeup(metrics):
    """Generates a technical summary for INR only."""
    print("Generating technical write-up...")
    start_time = time.time()
    writeup = ["Key Financial Indicators (INR):"]

    for key, values in metrics.items():
        avg = sum(values) / len(values) if values else 0
        writeup.append(f"- {key}: ₹{avg:,.0f} (based on {len(values)} mentions)")

    writeup.append("Analysis: Check if these numbers align with public claims—mismatches signal red flags.")
    print(f"Write-up generated in {time.time() - start_time:.2f} seconds")
    return "\n".join(writeup)

# Step 5: Visualizations (Rupees Only)
def create_visualizations(entities, metrics):
    """Generates visualizations for INR only."""
    print("Creating visualizations...")
    try:
        start_time = time.time()
        try:
            import seaborn as sns
            plt.style.use("ggplot")
        except ImportError:
            print("Seaborn not available—using default matplotlib style")
            plt.style.use("default")

        # Bar Chart: Entity Counts
        plt.figure(figsize=(10, 5))
        labels = ["Companies", "Money Mentions", "Dates"]
        counts = [len(entities["ORG"]), len(entities["MONEY"]), len(entities["DATE"])]
        plt.bar(labels, counts, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        plt.title("Key Elements in the Financial Statement")
        plt.ylabel("Count")
        plt.show()

        # Pie Chart: INR Metrics Breakdown
        if metrics:
            plt.figure(figsize=(8, 8))
            metric_labels = list(metrics.keys())
            metric_values = [sum(values) for values in metrics.values()]
            plt.pie(metric_values, labels=metric_labels, autopct="%1.1f%%", colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"])
            plt.title("INR Financial Metrics Breakdown (₹)")
            plt.show()

        print(f"Visualizations created in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Visualization error: {e}")

# Step 6: Main Application
def main():
    """Runs the full pipeline with debugging for INR only."""
    print("Financial Statement Simplifier v1.0 (Colab INR-Only Edition)\n")
    print("Ensure 'financial_report.pdf' is in your Google Drive root folder.\n")

    # Extract text
    text = extract_text()
    if not text:
        print("Stopping: No text extracted—check file path or contents!")
        return

    print("Sample Text (first 500 chars):")
    print(text[:500] + "\n")

    # Extract data
    entities, metrics = extract_financial_data(text)
    print("Extracted Entities:")
    for key, value in entities.items():
        print(f"{key}: {value[:5]} {'...' if len(value) > 5 else ''}")
    print("\nExtracted Metrics (INR):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print()

    # Simplify and generate story
    simplified, story = simplify_and_story(text, entities, metrics)
    print("Simplified Summary:")
    print(simplified + "\n")
    print("Layman’s Story:")
    print(story + "\n")

    # Technical write-up
    writeup = technical_writeup(metrics)
    print("Technical Write-Up:")
    print(writeup + "\n")

    # Visualizations
    create_visualizations(entities, metrics)

# Run the application
if __name__ == "__main__":
    main()
