from typing import Annotated
import re
from pydantic import BaseModel, Field, field_validator
from utils import EMPLOYMENT_TYPES, EXPERIENCE_LEVELS, EDUCATION_LEVELS

class JobPostingRequest(BaseModel):
    """
    Schema for validating incoming job posting predictions.
    Ensures data integrity before passing it to the ML pipeline.
    """

    title:           Annotated[str, Field(default="Unknown", max_length=200,  examples=["Machine Learning Engineer"])]
    location:        Annotated[str, Field(default="unknown", max_length=200,  examples=["Amman, Jordan"])]
    department:      Annotated[str, Field(default="Unknown", max_length=100,  examples=["Data Science"])]
    salary_range:    Annotated[str, Field(default="Missing", max_length=50,   examples=["1500-2500"])]
    company_profile: Annotated[str, Field(default="missing", max_length=5000, examples=["AI tech company..."])]
    description:     Annotated[str, Field(default="Unknown", max_length=5000, examples=["Looking for an ML engineer."])]
    requirements:    Annotated[str, Field(default="Unknown", max_length=5000, examples=["Python, FastAPI, Docker."])]
    benefits:        Annotated[str, Field(default="Unknown", max_length=2000, examples=["Health insurance."])]

    # Booleans are cleaner for APIs; they will be converted to 0/1 in the preprocessing pipeline
    telecommuting:    Annotated[bool, Field(default=False)]
    has_company_logo: Annotated[bool, Field(default=False)]
    has_questions:    Annotated[bool, Field(default=False)]

    # Link utils lists to Swagger UI for dropdown menus
    employment_type:     Annotated[str, Field(default="Unknown", json_schema_extra={"enum": EMPLOYMENT_TYPES})]
    required_experience: Annotated[str, Field(default="Unknown", json_schema_extra={"enum": EXPERIENCE_LEVELS})]
    required_education:  Annotated[str, Field(default="Unknown", json_schema_extra={"enum": EDUCATION_LEVELS})]

    industry: Annotated[str, Field(default="Unknown", max_length=200, examples=["Information Technology"])]
    function: Annotated[str, Field(default="Unknown", max_length=200, examples=["Engineering"])]

    # --- Validators ---

    @field_validator('title', 'location', 'department', 'company_profile',
                     'description', 'requirements', 'benefits', 'industry', 'function',
                     mode='before')
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        """Reject strings that are only whitespace."""
        if isinstance(value, str) and value.strip() == '':
            return 'Unknown'
        return value

    @field_validator('salary_range', mode='before')
    @classmethod
    def validate_salary_format(cls, value: str) -> str:
        """
        Accept: 'Missing', empty string, or 'number-number' format.
        Reject: random text that isn't a salary range.
        """
        if not value or value.strip() in ('', 'Missing', '0-0'):
            return 'Missing'
        # Allow numeric range like "1000-2000" or "1,000-2,000"
        cleaned = value.replace(',', '').strip()
        if re.match(r'^\d+(\.\d+)?-\d+(\.\d+)?$', cleaned):
            return cleaned
        # Anything else (text, single number, etc.) is treated as dirty/missing
        return value  # let preprocess.py handle it via is_dirty_salary flag

    @field_validator('employment_type')
    @classmethod
    def validate_employment(cls, value: str) -> str:
        if value not in EMPLOYMENT_TYPES:
            raise ValueError(f"Invalid employment type. Must be one of {EMPLOYMENT_TYPES}")
        return value

    @field_validator('required_experience')
    @classmethod
    def validate_experience(cls, value: str) -> str:
        if value not in EXPERIENCE_LEVELS:
            raise ValueError(f"Invalid experience level. Must be one of {EXPERIENCE_LEVELS}")
        return value

    @field_validator('required_education')
    @classmethod
    def validate_education(cls, value: str) -> str:
        if value not in EDUCATION_LEVELS:
            raise ValueError(f"Invalid education level. Must be one of {EDUCATION_LEVELS}")
        return value