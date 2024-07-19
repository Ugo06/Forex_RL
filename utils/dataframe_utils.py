import pandas as pd
import numpy as np

def missing_values_gaps(df:pd.DataFrame, column:str)->list:
    """
    This function takes a DataFrame and the name of a column, and returns a list containing
    the lengths of the sequences of missing values in that column.
    
    :param df: pandas DataFrame
    :param column: Name of the column to analyze
    :return: List of lengths of sequences of missing values
    """
    is_na = df[column].isna()
    
    gaps = []
    gap_length = 0

    for na in is_na:
        if na:
            gap_length += 1
        else:
            if gap_length > 0:
                gaps.append(gap_length)
                gap_length = 0
    
    if gap_length > 0:
        gaps.append(gap_length)
    
    return gaps

def filter_columns_by_gap(df:pd.DataFrame, max_gap:int)->pd.DataFrame:
    """
    Filters the columns of a DataFrame, keeping only those whose sequences of
    missing values do not contain any element exceeding the max_gap.
    
    :param df: pandas DataFrame
    :param max_gap: Maximum allowed length for a sequence of missing values
    :return: Filtered DataFrame
    """

    columns_to_keep = []
    
    for column in df.columns:
        gaps = missing_values_gaps(df, column)
        if all(gap <= max_gap for gap in gaps):
            columns_to_keep.append(column)
    
    return df[columns_to_keep]

