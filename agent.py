import argparse
import os
import pandas as pd
import pdfplumber
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import subprocess
import sys
import numpy as np
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()


# --- State Definition ---
class AgentState(TypedDict):
    """
    Represents the state of our agent.
    """

    target_bank: str
    pdf_path: str
    csv_path: str
    parser_code: str
    test_result: str
    error_message: str
    attempts: int


# --- Node Definitions ---


def plan_step(state: AgentState) -> AgentState:
    """
    Plans the process of generating the parser.
    """
    print("---PLANNING---")
    return state


def code_generation_step(state: AgentState) -> AgentState:
    """
    Generates the Python code for the parser using an LLM.
    """
    print("---GENERATING CODE---")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    csv_header = ",".join(pd.read_csv(state["csv_path"]).columns)

    prompt = f"""
    You are an expert Python programmer specializing in data extraction from PDFs.
    Your task is to write a single Python function `parse(pdf_path: str) -> pd.DataFrame` that extracts transaction data from a PDF bank statement.

    **Function Requirements:**
    1.  It must accept one argument: `pdf_path` (a string).
    2.  It must return a pandas DataFrame with the exact columns: {csv_header}
    3.  The function must use the `pdfplumber` library.

    **Implementation Guide:**
    -   Open the PDF using `pdfplumber.open(pdf_path)`.
    -   Iterate through each page in `pdf.pages`.
    -   On each page, use `page.extract_table()` to find the transaction table.
    -   The first row of the extracted table is likely the header; it should be skipped.
    -   Aggregate all transaction rows from all pages into a single list.
    -   Create a pandas DataFrame from the aggregated list of rows.
    -   Name the columns of the DataFrame exactly as follows: ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'].
    -   The 'Debit Amt' and 'Credit Amt' columns might contain empty strings or None. Replace these with `numpy.nan`.
    -   Convert 'Debit Amt', 'Credit Amt', and 'Balance' columns to a numeric type (float).

    Do not include any markdown formatting (like ```python) in your response. Output only the raw Python code for the function.
    """

    messages = [
        SystemMessage(
            content="You are a Python code generation assistant for PDF parsing."
        ),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    generated_code = (
        response.content.strip().replace("```python", "").replace("```", "")
    )

    state["parser_code"] = generated_code
    return state


def test_and_save_step(state: AgentState) -> AgentState:
    """
    Tests the generated parser code using pytest and saves it if the tests pass.
    """
    print("---TESTING AND SAVING WITH PYTEST---")
    parser_path = os.path.join("custom_parsers", f"{state['target_bank']}_parser.py")

    os.makedirs(os.path.dirname(parser_path), exist_ok=True)

    full_code = (
        "import pandas as pd\nimport pdfplumber\nimport numpy as np\n\n"
        + state["parser_code"]
    )

    with open(parser_path, "w") as f:
        f.write(full_code)

    try:
        # Execute pytest on the test file
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "test_parser.py"],
            capture_output=True,
            text=True,
        )

        # Pytest returns exit code 0 for success
        if result.returncode == 0:
            state["test_result"] = "success"
            print(f"---PYTEST PASSED---\n{result.stdout}")
        else:
            state["test_result"] = "failure"
            # Combine stdout and stderr for a complete error message
            error_output = result.stdout + "\n" + result.stderr
            state["error_message"] = error_output
            print(f"---PYTEST FAILED---\n{error_output}")

    except Exception as e:
        state["test_result"] = "failure"
        state["error_message"] = str(e)
        print(f"---TESTING FAILED WITH EXCEPTION---\n{e}")

    state["attempts"] = state.get("attempts", 0) + 1
    return state


# --- Conditional Logic ---
def should_continue(state: AgentState):
    """
    Determines whether to continue to the self-correction loop or end.
    """
    if state["test_result"] == "success":
        return "end"
    if state["attempts"] >= 3:
        print("---MAX ATTEMPTS REACHED---")
        return "end"
    return "continue"


def self_correction_step(state: AgentState) -> AgentState:
    """
    Attempts to correct the generated code based on the pytest error message.
    """
    print("---SELF-CORRECTING---")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    prompt = f"""
    The previously generated Python parser failed the pytest validation.

    **Original Code:**
    ```python
    {state['parser_code']}
    ```

    **Pytest Error Message:**
    ```
    {state['error_message']}
    ```

    **Instructions for Correction:**
    Please analyze the pytest error and the original code. Provide a corrected version of the `parse` function.
    -   The error message from `pandas.testing.assert_frame_equal` is very detailed. Pay close attention to mismatched values, types, or column names.
    -   Ensure you are correctly skipping the header row from the table data.
    -   Verify that `Debit Amt`, `Credit Amt`, and `Balance` are correctly converted to numeric types, and that empty values become `numpy.nan`.
    
    Only output the raw Python code for the corrected function, without any explanations or markdown.
    """

    messages = [
        SystemMessage(
            content="You are a Python code correction assistant for PDF parsing."
        ),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    corrected_code = (
        response.content.strip().replace("```python", "").replace("```", "")
    )

    state["parser_code"] = corrected_code
    print("---GENERATED CORRECTED CODE---")
    return state


# --- Main Agent Execution ---
def main(target_bank: str):
    """
    Main function to run the agent.
    """
    pdf_path = f"data/{target_bank}/{target_bank} sample.pdf"
    csv_path = f"data/{target_bank}/result.csv"

    if not os.path.exists(pdf_path) or not os.path.exists(csv_path):
        print(f"Error: Data files not found for target '{target_bank}'")
        return

    workflow = StateGraph(AgentState)
    workflow.add_node("plan", plan_step)
    workflow.add_node("generate_code", code_generation_step)
    workflow.add_node("test_and_save", test_and_save_step)
    workflow.add_node("self_correct", self_correction_step)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "generate_code")
    workflow.add_edge("generate_code", "test_and_save")
    workflow.add_conditional_edges(
        "test_and_save",
        should_continue,
        {
            "continue": "self_correct",
            "end": END,
        },
    )
    workflow.add_edge("self_correct", "test_and_save")

    app = workflow.compile()
    initial_state = AgentState(
        target_bank=target_bank,
        pdf_path=pdf_path,
        csv_path=csv_path,
        parser_code="",
        test_result="",
        error_message="",
        attempts=0,
    )

    final_state = app.invoke(initial_state)

    if final_state["test_result"] == "success":
        print(f"\nSuccessfully generated and tested parser for {target_bank}!")
        print(f"Parser saved to: custom_parsers/{target_bank}_parser.py")
    else:
        print(
            f"\nFailed to generate a working parser for {target_bank} after {final_state['attempts']} attempts."
        )
        print("Final error message:")
        print(final_state.get("error_message", "No error message captured."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agent-as-Coder for Bank Statement Parsing"
    )
    parser.add_argument(
        "--target", type=str, required=True, help="The target bank (e.g., 'icici')"
    )
    args = parser.parse_args()

    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.")
    else:
        # Make sure you have pytest installed: pip install pytest
        main(args.target)
