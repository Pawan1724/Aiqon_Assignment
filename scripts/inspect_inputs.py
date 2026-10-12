"""Inspect the assignment PDF and Excel dataset."""
import sys
import os

# Read the PDF
try:
    from PyPDF2 import PdfReader
    reader = PdfReader(r"")
    print(f"=== ASSIGNMENT PDF: {len(reader.pages)} pages ===")
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        print(f"\n--- Page {i+1} ---")
        print(text)
except Exception as e:
    print(f"PDF read error: {e}")

print("\n" + "="*80)

# Read the Excel dataset
import pandas as pd
import numpy as np

xlsx_path = r"c:\Users\joker\Desktop\test2\logistics-shipment-delay-prediction.xlsx"
xls = pd.ExcelFile(xlsx_path)
print(f"\n=== EXCEL FILE: {xlsx_path} ===")
print(f"Sheet names: {xls.sheet_names}")

for sheet in xls.sheet_names:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    print(f"\n--- Sheet: '{sheet}' ---")
    print(f"Shape: {df.shape}")
    print(f"\nColumn names ({len(df.columns)}):")
    for col in df.columns:
        print(f"  {col}: dtype={df[col].dtype}, nulls={df[col].isnull().sum()}, unique={df[col].nunique()}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    print(f"\nLast 3 rows:")
    print(df.tail(3).to_string())
    print(f"\nDescribe (numeric):")
    print(df.describe().to_string())
    print(f"\nDescribe (non-numeric):")
    print(df.describe(include='object').to_string())
    print(f"\nMissing values:")
    print(df.isnull().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    
    # Check for shipment ID column
    id_cols = [c for c in df.columns if 'id' in c.lower() or 'shipment' in c.lower()]
    if id_cols:
        for ic in id_cols:
            print(f"\n{ic}: total={len(df)}, unique={df[ic].nunique()}, duplicated={df[ic].duplicated().sum()}")
    
    # Check date columns
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date','time','arrival','departure','eta'])]
    if date_cols:
        print(f"\nDate-like columns: {date_cols}")
        for dc in date_cols:
            print(f"  {dc}: sample values = {df[dc].head(3).tolist()}")
    
    # Check delay columns
    delay_cols = [c for c in df.columns if 'delay' in c.lower()]
    if delay_cols:
        print(f"\nDelay columns: {delay_cols}")
        for dc in delay_cols:
            print(f"  {dc}: nulls={df[dc].isnull().sum()}, describe={df[dc].describe().to_dict() if df[dc].dtype in ['float64','int64'] else 'categorical'}")
