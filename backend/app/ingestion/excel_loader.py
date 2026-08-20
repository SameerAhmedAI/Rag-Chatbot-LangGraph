"""
Excel document loader (concrete Strategy).
Converts each sheet into readable row-based text chunks so the LLM
can reason over tabular data without needing a separate SQL layer.
"""

from langchain_core.documents import Document
import pandas as pd

from app.ingestion.base_loader import DocumentLoader


class ExcelLoader(DocumentLoader):
    """
    Loads .xlsx / .xls files. Each sheet becomes one or more Documents:
    a header/summary line describing the columns, followed by row-by-row
    text in "column: value" format, batched in groups so each Document
    stays a reasonable chunk size.
    """

    ROWS_PER_CHUNK = 20

    @property
    def file_type(self) -> str:
        return "excel"

    def load(self, file_path: str, source_name: str) -> list[Document]:
        documents: list[Document] = []

        sheets = pd.read_excel(file_path, sheet_name=None)  # dict of {sheet_name: DataFrame}

        for sheet_name, df in sheets.items():
            if df.empty:
                continue

            df = df.fillna("")
            columns = list(df.columns)

            for start in range(0, len(df), self.ROWS_PER_CHUNK):
                chunk_df = df.iloc[start:start + self.ROWS_PER_CHUNK]

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
                            "rows": f"{start + 1}-{min(start + self.ROWS_PER_CHUNK, len(df))}",
                            "file_type": self.file_type,
                        },
                    )
                )

        return documents