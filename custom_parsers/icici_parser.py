import pandas as pd
import pdfplumber
import numpy as np

import pdfplumber
import pandas as pd
import numpy as np

def parse(pdf_path: str) -> pd.DataFrame:
    pdf = pdfplumber.open(pdf_path)
    all_rows = []
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            header = table[0]
            rows = table[1:]
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if len(df.columns) == 5:
        df.columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
    elif len(df.columns) == 4:
        df.columns = ['Date', 'Description', 'Debit Amt', 'Balance']
        df['Credit Amt'] = np.nan
    else:
        raise ValueError("Unexpected number of columns in the extracted table.")


    df['Debit Amt'] = df['Debit Amt'].replace('', np.nan).astype(float)
    df['Credit Amt'] = df['Credit Amt'].replace('', np.nan).astype(float)
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')

    return df