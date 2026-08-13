"""
Excel document loader (.xlsx / .xls).
Converts each sheet into readable row-based text chunks so the LLM
can reason over tabular data without needing a separate SQL layer.
"""

from langchain_core.documents import Document
import pandas as pd


def load_excel(file_path: str, source_name: str) -> list[Document]:
    """
    Load an Excel file. Each sheet becomes one or more Documents:
    - A header/summary line describing the columns
    - Row-by-row text in "column: value" format, batched in groups
      so each Document stays a reasonable chunk size.
    """
    documents: list[Document] = []

    sheets = pd.read_excel(file_path, sheet_name=None)  # dict of {sheet_name: DataFrame}

    for sheet_name, df in sheets.items():
        if df.empty:
            continue

        df = df.fillna("")
        columns = list(df.columns)

        rows_per_chunk = 20
        for start in range(0, len(df), rows_per_chunk):
            chunk_df = df.iloc[start:start + rows_per_chunk]

            lines = [f"Sheet: {sheet_name} | Columns: {', '.join(str(c) for c in columns)}"]
            for _, row in chunk_df.iterrows():
                row_desc = "; ".join(f"{col}: {row[col]}" for col in columns)
                lines.append(row_desc)

            text = "\n".join(lines)

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "sheet": sheet_name,
                        "rows": f"{start + 1}-{min(start + rows_per_chunk, len(df))}",
                        "file_type": "excel",
                    },
                )
            )

    return documents