"""Utility to fetch prompts from Google Sheets by PROMPT_ID.

Supports sheets for each pipeline stage:
- KG_EXTRACTION_PROMPT: system/user prompts for KG extraction
- QUESTION_TO_CYPHER_PROMPT: prompts for NL → Cypher
- QUERY_RESULT_TO_ANSWER_PROMPT: prompts for Cypher results → NL answer
"""

import os
import logging
from typing import Dict, Optional

import pandas as pd
from modules.google_sheets_utils import GoogleUtil

logger = logging.getLogger(__name__)


def _get_google_util() -> GoogleUtil:
    """Create GoogleUtil from environment variables."""
    client_email = os.getenv("GOOGLE_SHEETS_CLIENT_EMAIL", "")
    private_key = os.getenv("GOOGLE_SHEETS_PRIVATE_KEY", "")
    if not client_email or not private_key:
        raise ValueError(
            "GOOGLE_SHEETS_CLIENT_EMAIL and GOOGLE_SHEETS_PRIVATE_KEY must be set in .env"
        )
    return GoogleUtil(private_key=private_key, client_email=client_email)


def fetch_prompt(
    sheet_name: str,
    prompt_id: str,
    spreadsheet_id: Optional[str] = None,
) -> Dict[str, str]:
    """Fetch a prompt row from Google Sheets by PROMPT_ID.

    Args:
        sheet_name: Worksheet name (e.g., "KG_EXTRACTION_PROMPT")
        prompt_id: Value to match in the PROMPT_ID column
        spreadsheet_id: Google Spreadsheet ID. If None, reads from GOOGLE_SPREADSHEET_ID env var.

    Returns:
        Dict with all columns from the matched row.
        For KG_EXTRACTION_PROMPT: {"PROMPT_ID": ..., "SYSTEM_PROMPT": ..., "USER_PROMPT": ...}

    Raises:
        ValueError: If PROMPT_ID not found or sheet is empty.
    """
    spreadsheet_id = spreadsheet_id or os.getenv("GOOGLE_SPREADSHEET_ID", "")
    if not spreadsheet_id:
        raise ValueError("GOOGLE_SPREADSHEET_ID must be set in .env or passed as argument")

    gu = _get_google_util()
    logger.info(f"Fetching prompt '{prompt_id}' from sheet '{sheet_name}'...")

    df = gu.load_dataframe_from_sheet(spreadsheet_id, sheet_name)

    if "PROMPT_ID" not in df.columns:
        raise ValueError(f"Sheet '{sheet_name}' must have a 'PROMPT_ID' column. Found: {list(df.columns)}")

    match = df[df["PROMPT_ID"] == prompt_id]
    if match.empty:
        available = df["PROMPT_ID"].tolist()
        raise ValueError(
            f"PROMPT_ID '{prompt_id}' not found in sheet '{sheet_name}'. "
            f"Available IDs: {available}"
        )

    row = match.iloc[0].to_dict()
    logger.info(f"Loaded prompt '{prompt_id}': {list(row.keys())}")
    return row


def fetch_kg_extraction_prompt(prompt_id: str) -> Dict[str, str]:
    """Shortcut to fetch KG extraction prompts.

    Returns:
        Dict with keys: PROMPT_ID, SYSTEM_PROMPT, USER_PROMPT
    """
    return fetch_prompt("KG_EXTRACTION_PROMPT", prompt_id)


def fetch_question_to_cypher_prompt(prompt_id: str) -> Dict[str, str]:
    """Shortcut to fetch NL → Cypher prompts.

    Returns:
        Dict with keys: PROMPT_ID, SYSTEM_PROMPT, USER_PROMPT
    """
    return fetch_prompt("QUESTION_TO_CYPHER_QUERY_PROMPT_TEMPLATE", prompt_id)


def fetch_query_result_to_answer_prompt(prompt_id: str) -> Dict[str, str]:
    """Shortcut to fetch Cypher results → NL answer prompts.

    Returns:
        Dict with keys: PROMPT_ID, SYSTEM_PROMPT, USER_PROMPT
    """
    return fetch_prompt("QUERY_RESULT_TO_ANSWER_PROMPT", prompt_id)
