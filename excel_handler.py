import pandas as pd
import io
from openpyxl.utils import get_column_letter
import re
import logging

logger = logging.getLogger(__name__)

class ExcelHandler:
    def export_to_excel(self, df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sanitize column names
            df = df.copy()
            df.columns = [self.sanitize_column_name(col) for col in df.columns]
            df.to_excel(writer, sheet_name='IFC Properties', index=True)
            # Adjust column widths
            worksheet = writer.sheets['IFC Properties']
            for idx, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col))
                worksheet.column_dimensions[get_column_letter(idx + 2)].width = max_len + 2
        return output.getvalue()

    def sanitize_column_name(self, name):
        # Remove invalid characters, but keep the dot separator
        name = re.sub(r'[^\w\s.-]', '', str(name))
        # Replace spaces with underscores
        name = name.replace(' ', '_')
        # Ensure the name starts with a letter or underscore
        if not name[0].isalpha() and name[0] != '_':
            name = '_' + name
        # Truncate name if it's too long (Excel has a 255 character limit)
        return name[:255]

    def import_from_excel(self, excel_file):
        try:
            logger.info(f"Starting Excel import for file: {excel_file.name}")
            df = pd.read_excel(excel_file, index_col=0)
            logger.info(f"Excel file read successfully. DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            logger.info(f"DataFrame index: {df.index.tolist()}")
            logger.info(f"Sample data:\n{df.head().to_string()}")
            return df
        except Exception as e:
            logger.error(f"Error importing Excel file: {str(e)}", exc_info=True)
            return None
