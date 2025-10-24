"""
Schemas y modelos de datos usando Pydantic
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# ============= ENUMS =============

class DatasetStatus(str, Enum):
    UPLOADED = "uploaded"
    CLEANING = "cleaning"
    CLEANED = "cleaned"
    TRAINING = "training"
    TRAINED = "trained"
    ERROR = "error"

class ModelType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    NEURAL_NETWORK = "neural_network"

class CleaningAction(str, Enum):
    REMOVE_DUPLICATES = "remove_duplicates"
    FILL_NULLS_MEAN = "fill_nulls_mean"
    FILL_NULLS_MEDIAN = "fill_nulls_median"
    FILL_NULLS_MODE = "fill_nulls_mode"
    DROP_NULLS = "drop_nulls"
    NORMALIZE = "normalize"
    STANDARDIZE = "standardize"
    ENCODE_CATEGORICAL = "encode_categorical"
    REMOVE_OUTLIERS = "remove_outliers"

# ============= REQUEST MODELS =============

class UploadResponse(BaseModel):
    success: bool
    message: str
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    preview: Optional[List[Dict[str, Any]]] = None
    column_info: Optional[Dict[str, Any]] = None

class CleaningRequest(BaseModel):
    file_id: str
    actions: List[CleaningAction]
    target_column: Optional[str] = None
    fill_value: Optional[Any] = None
    encoding_method: Optional[str] = "label"  # label, onehot
    outlier_method: Optional[str] = "iqr"  # iqr, zscore

class CleaningResponse(BaseModel):
    success: bool
    message: str
    file_id: Optional[str] = None
    original_rows: Optional[int] = None
    cleaned_rows: Optional[int] = None
    actions_applied: Optional[List[str]] = None
    preview: Optional[List[Dict[str, Any]]] = None
    statistics: Optional[Dict[str, Any]] = None

class TrainingRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    file_id: str
    model_type: ModelType
    target_column: str
    feature_columns: List[str]
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    algorithm: Optional[str] = None  # linear_regression, random_forest, svm, neural_network, etc.
    hyperparameters: Optional[Dict[str, Any]] = None

class TrainingResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    success: bool
    message: str
    model_id: Optional[str] = None
    model_type: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time: Optional[float] = None
    predictions_sample: Optional[List[Any]] = None

class AgentRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    file_id: Optional[str] = None

class AgentResponse(BaseModel):
    success: bool
    message: str
    suggestions: Optional[List[str]] = None
    next_step: Optional[str] = None
    warning: Optional[str] = None

class DatasetInfo(BaseModel):
    file_id: str
    file_name: str
    status: DatasetStatus
    rows: int
    columns: int
    created_at: datetime
    updated_at: datetime
    column_types: Optional[Dict[str, str]] = None
    missing_values: Optional[Dict[str, int]] = None

class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_id: str
    model_name: str
    model_type: ModelType
    dataset_id: str
    created_at: datetime
    metrics: Dict[str, float]
    status: str

class DashboardStats(BaseModel):
    total_datasets: int
    total_models: int
    total_cleaned_datasets: int
    recent_datasets: List[DatasetInfo]
    recent_models: List[ModelInfo]
    storage_used_mb: float

# ============= DATABASE MODELS =============

class DatasetDB(BaseModel):
    id: Optional[str] = None
    file_name: str
    file_path: str
    status: str
    rows: int
    columns: int
    column_types: Optional[Dict[str, str]] = None
    missing_values: Optional[Dict[str, int]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ModelDB(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    id: Optional[str] = None
    model_name: str
    model_type: str
    dataset_id: str
    model_path: str
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    status: str = "trained"

class CleaningLogDB(BaseModel):
    id: Optional[str] = None
    dataset_id: str
    actions: List[str]
    original_rows: int
    cleaned_rows: int
    statistics: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

