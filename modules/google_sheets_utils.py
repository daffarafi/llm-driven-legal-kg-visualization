"""Utility class for Google-related logic.

It contains utils and logic to write to Google Spreadsheet.

Authors:
    Komang Elang Surya Prawira (komang.e.s.prawira@gdplabs.id)
    Alfan Dinda Rahmawan (alfan.d.rahmawan@gdplabs.id)

Reviewers:
    Pray Somaldo (pray.somaldo@gdplabs.id)

References:
    [1] https://github.com/GDP-ADMIN/gen-ai-veriwise/blob/cli-script/module/utility/google.py
"""

import logging
import time
from dataclasses import dataclass
from random import uniform
from typing import Any, Dict, List, Optional

import gspread
import pandas as pd
from google.oauth2 import service_account  # pylint: disable=E0611
from gspread import Client, Spreadsheet, Worksheet
from gspread.cell import Cell
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BatchWriteResult:
    """Class to store results of batch write operations"""
    successful_rows: int
    failed_rows: int
    errors: List[Dict[str, Any]]

class GoogleSheetsWriter:
    """Class to handle writing data to Google Sheets with retry logic and batch processing"""

    def __init__(
        self,
        google_util: Any,
        sheet_id: str,
        worksheet_name: str,
        batch_size: int = 10,
        max_retries: int = 5,
        max_delay: float = 32.0,
        batch_delay: float = 2.0
    ):
        """Initialize GoogleSheetsWriter.

        Args:
            google_util: GoogleUtil instance for sheet operations
            sheet_id (str): Google Sheet ID
            worksheet_name (str): Name of worksheet
            batch_size (int, optional): Number of rows to process in each batch. Defaults to 10.
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.
            max_delay (float, optional): Maximum delay between retries in seconds. Defaults to 32.0.
            batch_delay (float, optional): Delay between batches in seconds. Defaults to 2.0.
        """
        self.google_util = google_util
        self.sheet_id = sheet_id
        self.worksheet_name = worksheet_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.max_delay = max_delay
        self.batch_delay = batch_delay

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate delay time using exponential backoff with jitter.

        Args:
            attempt (int): Current attempt number (starting from 1)

        Returns:
            float: Time to wait in seconds
        """
        return min(self.max_delay, (2 ** attempt) + uniform(0, 1))

    def _write_single_row(self, row_data: Dict[str, Any]) -> Optional[Exception]:
        """Write a single row to spreadsheet with retry logic.

        Args:
            row_data (Dict[str, Any]): Row data to write

        Returns:
            Optional[Exception]: Exception if write failed after all retries, None if successful
        """
        attempt = 0
        while attempt < self.max_retries:
            try:
                self.google_util.write_to_spreadsheet(
                    sheet_id=self.sheet_id,
                    worksheet_output_name=self.worksheet_name,
                    **row_data
                )
                return None
            except Exception as e:
                attempt += 1
                error_str = str(e)

                # Transient errors worth retrying
                is_retryable = any(code in error_str for code in ["429", "500", "502", "503", "504"])

                if attempt == self.max_retries:
                    logger.error(f"Failed after {self.max_retries} attempts for row: {row_data}")
                    return e

                if is_retryable:
                    delay = self._exponential_backoff(attempt)
                    logger.warning(f"Transient error ({error_str[:60]}). Waiting {delay:.2f}s before retry {attempt}/{self.max_retries}")
                    time.sleep(delay)
                else:
                    return e  # Non-retryable errors fail immediately

    def write_dataframe(self, df: pd.DataFrame, show_progress: bool = True) -> BatchWriteResult:
        """Write DataFrame to Google Sheets in batches.

        Args:
            df (pd.DataFrame): DataFrame containing rows to write
            show_progress (bool, optional): Whether to show progress bar. Defaults to True.

        Returns:
            BatchWriteResult: Results of the batch write operation
        """
        successful_rows = 0
        failed_rows = 0
        errors = []

        # Create batches
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size

        # Setup progress bar if requested
        batch_iterator = tqdm(range(0, len(df), self.batch_size)) if show_progress else range(0, len(df), self.batch_size)

        for i in batch_iterator:
            batch = df.iloc[i:i + self.batch_size]

            # Process each row in the batch
            for index, row in batch.iterrows():
                row_number = index + 1
                error = self._write_single_row(row.to_dict())

                if error is None:
                    successful_rows += 1
                    if show_progress:
                        logger.info(f"Successfully wrote row {row_number}/{len(df)}")
                else:
                    failed_rows += 1
                    errors.append({
                        "row_number": row_number,
                        "error": str(error),
                        "data": row.to_dict()
                    })
                    logger.error(f"Error writing row {row_number}: {str(error)}")

            # Delay between batches if not the last batch
            if i + self.batch_size < len(df):
                time.sleep(self.batch_delay)

        return BatchWriteResult(successful_rows, failed_rows, errors)



class GoogleUtil:
    """Utility class to contain Google-related logic.

    Attributes:
        private_key (str): The private key associated with the Google service account.
        client_email (str): The email address of the Google service account.
        GOOGLE_SHEETS_SCOPES (List[str]): Scopes for Google Sheets API access.
        logger (Logger): Logger instance for logging information and errors.

    Methods:
        write_to_spreadsheet(**kwargs): Write data to a specified Google Spreadsheet.
        _get_worksheet(client_email, private_key,
                       sheet_id, worksheet_name): Retrieve a worksheet object from a Google Spreadsheet.
        _get_client(info, scopes): Create an authorized Google client for API requests.
        _get_google_info(private_key, client_email): Construct service account information for authentication.
    """

    def __init__(self, private_key: str, client_email: str):
        """Initialize the GoogleUtil object with the required credentials.

        Args:
            private_key (str): The private key associated with the Google service account.
            client_email (str): The email address of the Google service account.
        """
        self.private_key: str = private_key
        self.client_email: str = client_email
        self.GOOGLE_SHEETS_SCOPES: List[str] = ["https://www.googleapis.com/auth/spreadsheets"]

    def retrieve_worksheet(self, sheet_id: str, worksheet_id: str) -> List[List[str]]:
        """Retrieve prompts from Google Spreadsheet.

        Args:
            sheet_id (str): The ID of the spreadsheet
            worksheet_id (str): The ID of the worksheet

        Returns:
            List[List[str]]: The list of prompts retrieved from the Google Spreadsheet.
        """
        worksheet: Worksheet = self._get_worksheet(self.client_email, self.private_key, sheet_id, worksheet_id)
        return worksheet.get_all_values()

    def _get_worksheet(self, client_email: str, private_key: str, sheet_id: str, worksheet_name: str) -> Worksheet:
        """Function to get worksheet object.

        Args:
            client_email (str): The email address of the Google service account.
            private_key (str): The private key associated with the service account.
            sheet_id (str): The ID of the Google Sheet.
            worksheet_name (str): The name of the worksheet within the Google Sheet.

        Returns:
            Worksheet: The worksheet object retrieved using the provided IDs and credentials.
        """
        client: Client = self._get_client(self._get_google_info(private_key, client_email), self.GOOGLE_SHEETS_SCOPES)
        sh: Spreadsheet = client.open_by_key(sheet_id)
        worksheet: Worksheet = sh.worksheet(worksheet_name)
        return worksheet

    @staticmethod
    def _get_client(info: Dict[str, Any], scopes: List[str]) -> Client:
        """Function to get Google client.

        Args:
            info (Dict[str, Any]): The service account information required for authentication.
            scopes (List[str]): The scopes defining access permissions.

        Returns:
            Client: The authorized Google client instance used for API requests.
        """
        credentials: service_account.Credentials = service_account.Credentials.from_service_account_info(info=info, scopes=scopes)
        client: Client = gspread.authorize(credentials)
        return client

    @staticmethod
    def _get_google_info(private_key: str, client_email: str) -> Dict[str, str]:
        """Function to get Google info.

        Args:
            private_key (str): The private key associated with the service account.
            client_email (str): The email address of the Google service account.

        Returns:
            Dict[str, str]: Service account information.
        """
        return {
            "type": "service_account",
            "private_key": private_key,
            "client_email": client_email,
            "client_id": "https://www.googleapis.com/auth/spreadsheets",
            "token_uri": "https://oauth2.googleapis.com/token",
        }

    def write_to_spreadsheet(self, sheet_id: str, worksheet_output_name: str, **kwargs: Dict[str, Any]):
        """Write to Google Spreadsheet.

        The parameters are passed as keyword arguments.

        Args:
            sheet_id (str): Sheet ID.
            worksheet_output_name (str): Sheet name.
        """
        worksheet: Worksheet = self._get_worksheet(self.client_email, self.private_key, sheet_id, worksheet_output_name)

        last_row: int = len(worksheet.get_all_values()) + 1

        cells: List[Cell] = []
        idx: int = 1
        for value in kwargs.values():
            cells.append(Cell(row=last_row, col=idx, value=str(value)))
            idx += 1

        worksheet.update_cells(cells)

    def load_dataframe_from_sheet(
        self,
        spreadsheet_id: str,
        sheet_name: str,
        header_row: int = 0,
        dtype: Optional[Dict[str, Any]] = None,
        na_values: Optional[List[str]] = None,
        parse_dates: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load data from a Google Sheet worksheet into a pandas DataFrame.

        Args:
            spreadsheet_id (str): The ID of the Google Spreadsheet.
            sheet_name (str): The name of the worksheet to retrieve data from.
            header_row (int, optional): Row index to use as column names (0-based). Defaults to 0.
            dtype (Dict[str, Any], optional): Dictionary of column data types. Defaults to None.
            na_values (List[str], optional): Additional strings to recognize as NA/NaN. Defaults to None.
            parse_dates (List[str], optional): List of column names to parse as dates. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the worksheet data.

        Raises:
            ValueError: If the worksheet is empty or contains only a header row.
        """
        try:
            logging.info(f"Retrieving worksheet '{sheet_name}' from spreadsheet ID '{spreadsheet_id}'...")
            rows = self.retrieve_worksheet(spreadsheet_id, sheet_name)

            if not rows:
                logging.warning(f"Worksheet '{sheet_name}' is empty.")
                raise ValueError(f"No data found in worksheet: {sheet_name}")

            if len(rows) <= header_row + 1:
                logging.warning(f"Worksheet '{sheet_name}' contains insufficient data rows.")
                raise ValueError(f"Insufficient data rows in worksheet: {sheet_name}")

            # Extract header and data based on header_row
            header = rows[header_row]
            data = rows[header_row + 1:]

            # Create DataFrame with specified options
            df = pd.DataFrame(data, columns=header)

            # Apply data type conversions if specified
            if dtype:
                df = df.astype({col: dtype[col] for col in dtype if col in df.columns})

            # Handle NA values
            if na_values:
                for col in df.columns:
                    df[col] = df[col].replace(na_values, pd.NA)

            # Parse dates if specified
            if parse_dates:
                for date_col in parse_dates:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

            logging.info(f"Successfully loaded {len(df)} rows from worksheet '{sheet_name}'.")
            return df

        except ValueError:
            # Re-raise specific ValueError exceptions
            raise
        except Exception as e:
            logging.error(f"Failed to load data from worksheet '{sheet_name}': {e}", exc_info=True)
            raise ValueError(f"Error loading worksheet '{sheet_name}': {str(e)}")