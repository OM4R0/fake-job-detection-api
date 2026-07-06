import re
import pandas as pd
import category_encoders as ce
from sklearn.feature_extraction.text import TfidfVectorizer

def process_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Extract min and max salary values and flag dirty/textual entries."""
    df['salary_clean'] = df['salary_range'].fillna('Missing')
    df['has_salary'] = df['salary_clean'].apply(lambda x: 0 if x in ['Missing', '0-0'] else 1)
    
    def check_if_dirty(val):
        if val == 'Missing': return 0
        if val == '0-0' or re.search(r'[a-zA-Z]', str(val)): return 1
        return 0
        
    df['is_dirty_salary'] = df['salary_clean'].apply(check_if_dirty)
    
    def extract_min_max(val):
        if val in ['Missing', '0-0'] or re.search(r'[a-zA-Z]', str(val)): return -1.0, -1.0
        try:
            parts = str(val).split('-')
            if len(parts) == 2: return float(parts[0]), float(parts[1])
        except: pass
        return -1.0, -1.0
        
    df['min_salary'], df['max_salary'] = zip(*df['salary_clean'].apply(extract_min_max))
    return df.drop(columns=['salary_range', 'salary_clean'])

def create_presence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary flags (0/1) for the presence of specific columns before filling nulls."""
    columns_to_check = ['department', 'company_profile', 'requirements', 'benefits', 'industry', 'function']
    for col in columns_to_check:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)
            
    if 'department' in df.columns:
        df = df.drop(columns=['department'])
    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize missing values across categorical and numeric columns."""
    fill_mapping = {
        'location': 'unknown', 'company_profile': 'missing', 'description': 'Unknown',
        'requirements': 'Unknown', 'benefits': 'Unknown', 'employment_type': 'Unknown',
        'industry': 'Unknown', 'function': 'Unknown', 'required_experience': -1, 'required_education': -1
    }
    for col, value in fill_mapping.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    return df

def create_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate character length for textual fields."""
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_columns:
        if col in df.columns:
            df[f'{col}_length'] = df[col].fillna('').astype(str).str.len()
    return df

def apply_ordinal_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Map ordinal categorical variables to numeric scales to preserve hierarchy."""
    experience_map = {
        'Internship': 1, 'Entry level': 2, 'Associate': 3, 
        'Mid-Senior level': 4, 'Director': 5, 'Executive': 6
    }
    education_map = {
        'Unspecified': 0, 'Some High School Coursework': 1, 'High School or equivalent': 2, 
        'Vocational': 3, 'Vocational - HS Diploma': 3, 'Vocational - Degree': 4, 
        'Some College Coursework Completed': 5, 'Associate Degree': 6, 
        "Bachelor's Degree": 7, "Master's Degree": 8, 'Professional': 9, 'Doctorate': 10
    }
    
    if 'required_experience' in df_train.columns:
        df_train['required_experience'] = df_train['required_experience'].map(experience_map).fillna(-1)
        df_test['required_experience'] = df_test['required_experience'].map(experience_map).fillna(-1)
        
    if 'required_education' in df_train.columns:
        df_train['required_education'] = df_train['required_education'].map(education_map).fillna(-1)
        df_test['required_education'] = df_test['required_education'].map(education_map).fillna(-1)
        
    return df_train, df_test

def apply_target_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame, y_train):
    """Use Target Encoding for high-cardinality columns to prevent memory bloat and curse of dimensionality."""
    encoder = ce.TargetEncoder(cols=['industry', 'location', 'function'], smoothing=10)
    df_train = encoder.fit_transform(df_train, y_train)
    df_test = encoder.transform(df_test)
    return df_train, df_test, encoder

def apply_ohe(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """One-Hot Encode low-cardinality nominal variables."""
    ohe_encoder = ce.OneHotEncoder(cols=['employment_type'], use_cat_names=True)
    df_train = ohe_encoder.fit_transform(df_train)
    df_test = ohe_encoder.transform(df_test)
    return df_train, df_test, ohe_encoder

def apply_tfidf(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Combine text columns and vectorize using TF-IDF to extract keyword importance."""
    TF_IDF_columns = ['company_profile', 'title', 'description', 'requirements', 'benefits']
    
    df_train[TF_IDF_columns] = df_train[TF_IDF_columns].fillna('')
    df_test[TF_IDF_columns] = df_test[TF_IDF_columns].fillna('')
    
    df_train['combined_text'] = df_train[TF_IDF_columns].apply(lambda x: ' '.join(x), axis=1)
    df_test['combined_text'] = df_test[TF_IDF_columns].apply(lambda x: ' '.join(x), axis=1)
    
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    
    train_tfidf = pd.DataFrame(tfidf.fit_transform(df_train['combined_text']).toarray(), index=df_train.index)
    test_tfidf = pd.DataFrame(tfidf.transform(df_test['combined_text']).toarray(), index=df_test.index)
    
    df_train = df_train.drop(columns=TF_IDF_columns + ['combined_text'])
    df_test = df_test.drop(columns=TF_IDF_columns + ['combined_text'])
    
    df_train = pd.concat([df_train, train_tfidf], axis=1)
    df_test = pd.concat([df_test, test_tfidf], axis=1)
    
    return df_train, df_test, tfidf