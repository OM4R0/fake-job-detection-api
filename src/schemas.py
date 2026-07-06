from typing import Annotated
from pydantic import BaseModel, Field, field_validator
from utils import EMPLOYMENT_TYPES, EXPERIENCE_LEVELS, EDUCATION_LEVELS

class JobPostingRequest(BaseModel):
    """
    Schema for validating incoming job posting predictions.
    Ensures data integrity before passing it to the ML pipeline.
    """
    
    title: Annotated[str, Field(default="Unknown", examples=["Machine Learning Engineer"])]
    location: Annotated[str, Field(default="unknown", examples=["Amman, Jordan"])]
    department: Annotated[str, Field(default="Unknown", examples=["Data Science"])]
    salary_range: Annotated[str, Field(default="Missing", examples=["1500-2500"])]
    company_profile: Annotated[str, Field(default="missing", examples=["AI tech company..."])]
    description: Annotated[str, Field(default="Unknown", examples=["Looking for an ML engineer."])]
    requirements: Annotated[str, Field(default="Unknown", examples=["Python, FastAPI, Docker."])]
    benefits: Annotated[str, Field(default="Unknown", examples=["Health insurance."])]
    
    # Booleans are cleaner for APIs; they will be converted to 0/1 in the preprocessing pipeline
    telecommuting: Annotated[bool, Field(default=False)]
    has_company_logo: Annotated[bool, Field(default=False)]
    has_questions: Annotated[bool, Field(default=False)]
    
    # Link utils lists to Swagger UI for dropdown menus
    employment_type: Annotated[str, Field(default="Unknown", json_schema_extra={"enum": EMPLOYMENT_TYPES})]
    required_experience: Annotated[str, Field(default="Unknown", json_schema_extra={"enum": EXPERIENCE_LEVELS})]
    required_education: Annotated[str, Field(default="Unknown", json_schema_extra={"enum": EDUCATION_LEVELS})]
    
    industry: Annotated[str, Field(default="Unknown", examples=["Information Technology"])]
    function: Annotated[str, Field(default="Unknown", examples=["Engineering"])]

    # --- Validators to enforce strict inputs based on utils lists ---

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