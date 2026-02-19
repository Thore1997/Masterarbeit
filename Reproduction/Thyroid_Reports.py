import pandas as pd
from fpdf import FPDF

# --- 1. Data Processing ---
file_path = 'data/thyroid0387.data'
df = pd.read_csv(file_path, sep=',', header=None, na_values='?')

column_mapping = {
    0: 'Age',
    17: 'TSH',
    19: 'T3',
    21: 'TT4',
    23: 'T4U',
    25: 'FTI',
    27: 'TBG'
}

df_selected = df[list(column_mapping.keys())].rename(columns=column_mapping)

# Prepare Statistics Data
stats_full = df_selected.describe().reset_index()

# Filter out the 25% and 75% rows
# The 'index' column contains the names of the statistics
stats = stats_full[~stats_full['index'].isin(['25%', '75%'])]

# Prepare Missing Values Data
missing_count = df_selected.isnull().sum().reset_index()
missing_count.columns = ['Feature', 'Missing Count']
missing_count['Percentage'] = (missing_count['Missing Count'] / len(df_selected)) * 100


# --- 2. Custom PDF Class ---
class ThyroidReport(FPDF):
    def __init__(self, report_title):
        super().__init__()
        self.report_title = report_title

    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, self.report_title, border=False, ln=True, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')


# --- 3. Generate PDF #1: Descriptive Statistics (Filtered) ---
pdf_stats = ThyroidReport("Report 1: Descriptive Statistics")
pdf_stats.add_page()
pdf_stats.set_font("Arial", size=10)
pdf_stats.cell(0, 10, f"Source Data: {file_path} (Quartiles Removed)", ln=True)

with pdf_stats.table(text_align="CENTER", width=190) as table:
    header = table.row()
    for col in stats.columns:
        header.cell(col)

    for _, row in stats.iterrows():
        data_row = table.row()
        for i, val in enumerate(row):
            # Format numbers to 2 decimal places, keep label column as string
            text = f"{val:.2f}" if isinstance(val, (float, int)) and i > 0 else str(val)
            data_row.cell(text)

pdf_stats.output("Thyroid_Statistics_Report.pdf")

# --- 4. Generate PDF #2: Missing Values Analysis ---
pdf_missing = ThyroidReport("Report 2: Missing Values Analysis")
pdf_missing.add_page()
pdf_missing.set_font("Arial", size=11)
pdf_missing.cell(0, 10, f"Total Dataset Observations: {len(df_selected)}", ln=True)
pdf_missing.ln(5)

with pdf_missing.table(text_align="CENTER", width=160) as table:
    header = table.row()
    for col in missing_count.columns:
        header.cell(col)

    for _, row in missing_count.iterrows():
        data_row = table.row()
        data_row.cell(str(row['Feature']))
        data_row.cell(str(row['Missing Count']))
        data_row.cell(f"{row['Percentage']:.2f}%")

pdf_missing.output("Thyroid_Missing_Values_Report.pdf")

print("Success! Reports generated with filtered statistics.")