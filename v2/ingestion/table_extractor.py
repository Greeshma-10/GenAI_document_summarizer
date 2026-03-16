# v2/ingestion/table_extractor.py

import camelot


def extract_tables(file_path):

    tables = camelot.read_pdf(file_path, pages="all")

    extracted_tables = []

    for table in tables:
        extracted_tables.append(table.df.to_string())

    return extracted_tables