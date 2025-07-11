"""
Schema-driven data handling for time series framework.
Eliminates hardcoded column names and provides configurable data validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""
    name: str
    dtype: str  # 'float', 'int', 'str', 'datetime'
    required: bool = True
    default_value: Any = None
    description: str = ""
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a single value against this column schema."""
        if pd.isna(value):
            if self.required and self.default_value is None:
                return False, f"Required column '{self.name}' has missing values"
            return True, None
            
        # Type validation
        if self.dtype == 'float':
            try:
                float(value)
                return True, None
            except (ValueError, TypeError):
                return False, f"Column '{self.name}' expects float but got {type(value)}"
        elif self.dtype == 'int':
            try:
                int(value)
                return True, None
            except (ValueError, TypeError):
                return False, f"Column '{self.name}' expects int but got {type(value)}"
        elif self.dtype == 'str':
            if not isinstance(value, str):
                return False, f"Column '{self.name}' expects str but got {type(value)}"
            return True, None
        elif self.dtype == 'datetime':
            if not isinstance(value, (pd.Timestamp, str)):
                return False, f"Column '{self.name}' expects datetime but got {type(value)}"
            return True, None
        else:
            return False, f"Unknown dtype '{self.dtype}' for column '{self.name}'"


class StudentIDStrategy:
    """Defines how to incorporate student ID as a feature."""
    
    def __init__(self, strategy_type: str = 'none', **kwargs):
        """
        Args:
            strategy_type: 'none', 'onehot', 'target_encoding', 'embeddings', 'mixed_effects'
            **kwargs: Strategy-specific parameters
        """
        self.strategy_type = strategy_type
        self.params = kwargs
        
    def get_additional_features(self, df: pd.DataFrame, student_column: str, target_column: str) -> pd.DataFrame:
        """
        Generate additional features based on student ID strategy.
        
        Args:
            df: Input dataframe
            student_column: Name of student ID column
            target_column: Name of target column
            
        Returns:
            DataFrame with additional student-based features
        """
        if self.strategy_type == 'none':
            return df
            
        elif self.strategy_type == 'onehot':
            # One-hot encode student IDs
            student_dummies = pd.get_dummies(df[student_column], prefix='student')
            return pd.concat([df, student_dummies], axis=1)
            
        elif self.strategy_type == 'target_encoding':
            # Add student-specific statistics
            return self._add_target_encoding_features(df, student_column, target_column)
            
        elif self.strategy_type == 'embeddings':
            # For neural networks - just add student ID as integer
            student_mapping = {student: idx for idx, student in enumerate(df[student_column].unique())}
            df['student_id_numeric'] = df[student_column].map(student_mapping)
            return df
            
        elif self.strategy_type == 'mixed_effects':
            # Mixed effects models handle this internally
            return df
            
        elif self.strategy_type == 'universal':
            # Universal approach: student ID is already in feature columns
            # Different adapters will handle it appropriately
            return df
            
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
    
    def _add_target_encoding_features(self, df: pd.DataFrame, student_column: str, target_column: str) -> pd.DataFrame:
        """Add target encoding features for students."""
        # Calculate student-specific statistics
        student_stats = df.groupby(student_column)[target_column].agg([
            'mean', 'std', 'count', 'median'
        ]).reset_index()
        student_stats.columns = [student_column, 'student_target_mean', 'student_target_std', 
                               'student_target_count', 'student_target_median']
        
        # Handle NaN values
        student_stats['student_target_std'] = student_stats['student_target_std'].fillna(0)
        
        # Add global statistics for new students
        global_mean = df[target_column].mean()
        global_std = df[target_column].std()
        
        student_stats['student_target_mean'] = student_stats['student_target_mean'].fillna(global_mean)
        student_stats['student_target_std'] = student_stats['student_target_std'].fillna(global_std)
        
        # Merge back to original dataframe
        df_with_stats = df.merge(student_stats, on=student_column, how='left')
        
        # Add additional derived features
        df_with_stats['student_target_deviation'] = (
            df_with_stats[target_column] - df_with_stats['student_target_mean']
        )
        
        return df_with_stats
    
    def get_additional_feature_columns(self, df: pd.DataFrame, student_column: str) -> List[str]:
        """Get list of additional feature column names this strategy will create."""
        if self.strategy_type == 'none':
            return []
        elif self.strategy_type == 'onehot':
            return [f'student_{student}' for student in df[student_column].unique()]
        elif self.strategy_type == 'target_encoding':
            return ['student_target_mean', 'student_target_std', 'student_target_count', 
                   'student_target_median', 'student_target_deviation']
        elif self.strategy_type == 'embeddings':
            return ['student_id_numeric']
        elif self.strategy_type == 'mixed_effects':
            return []
        elif self.strategy_type == 'universal':
            return []  # Student ID is already in the feature columns
        else:
            return []


@dataclass
class DataSchema:
    """Complete schema definition for time series data."""
    # Core column definitions
    student_column: str
    time_column: str
    target_column: str
    feature_columns: List[str]
    
    # Column schemas
    columns: Dict[str, ColumnSchema] = field(default_factory=dict)
    
    # Student ID integration strategy
    student_id_strategy: StudentIDStrategy = field(default_factory=lambda: StudentIDStrategy('none'))
    
    # Additional configuration
    time_format: Optional[str] = None  # e.g., '%Y-W%W' for week strings
    min_sequence_length: int = 2
    validation_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize column schemas if not provided."""
        if not self.columns:
            # Auto-generate basic schemas
            all_columns = (
                [self.student_column, self.time_column, self.target_column] + 
                self.feature_columns
            )
            for col in all_columns:
                if col not in self.columns:
                    # Infer dtype based on column name patterns
                    if 'id' in col.lower() or col == 'name':
                        dtype = 'str'
                    elif 'week' in col.lower() or 'date' in col.lower():
                        dtype = 'float'  # Week numbers are numeric
                    else:
                        dtype = 'float'  # Default for numeric features
                    
                    self.columns[col] = ColumnSchema(
                        name=col,
                        dtype=dtype,
                        required=True
                    )
    
    @classmethod
    def from_config(cls, config: Union[Dict, str, Path]) -> 'DataSchema':
        """Create schema from configuration dict or file."""
        if isinstance(config, (str, Path)):
            with open(config, 'r') as f:
                config = json.load(f)
        
        # Extract column schemas if provided
        columns = {}
        if 'columns' in config:
            for col_name, col_config in config['columns'].items():
                columns[col_name] = ColumnSchema(
                    name=col_name,
                    **col_config
                )
        
        return cls(
            student_column=config['student_column'],
            time_column=config['time_column'],
            target_column=config['target_column'],
            feature_columns=config['feature_columns'],
            columns=columns,
            time_format=config.get('time_format'),
            min_sequence_length=config.get('min_sequence_length', 2),
            validation_rules=config.get('validation_rules', {})
        )
    
    def to_config(self) -> Dict:
        """Export schema to configuration dict."""
        columns_config = {}
        for col_name, col_schema in self.columns.items():
            columns_config[col_name] = {
                'dtype': col_schema.dtype,
                'required': col_schema.required,
                'default_value': col_schema.default_value,
                'description': col_schema.description
            }
        
        return {
            'student_column': self.student_column,
            'time_column': self.time_column,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'columns': columns_config,
            'time_format': self.time_format,
            'min_sequence_length': self.min_sequence_length,
            'validation_rules': self.validation_rules
        }
    
    def get_all_columns(self) -> List[str]:
        """Get all column names used in the schema."""
        return list(set(
            [self.student_column, self.time_column, self.target_column] + 
            self.feature_columns
        ))
    
    def get_feature_indices(self) -> Dict[str, int]:
        """Get feature name to index mapping for array operations."""
        indices = {}
        for i, col in enumerate(self.feature_columns):
            indices[col] = i
        return indices


class DataValidator:
    """Validates datasets against a schema."""
    
    def __init__(self, schema: DataSchema):
        self.schema = schema
    
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate a dataframe against the schema.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required columns exist
        required_columns = self.schema.get_all_columns()
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Validate column types and values
        for col_name, col_schema in self.schema.columns.items():
            if col_name in df.columns:
                # Sample validation (check first 100 non-null values)
                sample = df[col_name].dropna().head(100)
                for value in sample:
                    is_valid, error = col_schema.validate(value)
                    if not is_valid:
                        issues.append(error)
                        break  # One error per column is enough
        
        # Check temporal consistency
        if self.schema.time_column in df.columns and self.schema.student_column in df.columns:
            temporal_issues = self._check_temporal_consistency(df)
            issues.extend(temporal_issues)
        
        # Check minimum sequence lengths
        if self.schema.student_column in df.columns:
            seq_length_issues = self._check_sequence_lengths(df)
            issues.extend(seq_length_issues)
        
        return len(issues) == 0, issues
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> List[str]:
        """Check for temporal consistency issues."""
        issues = []
        
        # Check for duplicate time entries per student
        duplicates = df.groupby([self.schema.student_column, self.schema.time_column]).size()
        duplicate_entries = duplicates[duplicates > 1]
        if len(duplicate_entries) > 0:
            issues.append(
                f"Found {len(duplicate_entries)} duplicate time entries for students"
            )
        
        return issues
    
    def _check_sequence_lengths(self, df: pd.DataFrame) -> List[str]:
        """Check that students have minimum required sequence length."""
        issues = []
        
        student_counts = df.groupby(self.schema.student_column).size()
        short_sequences = student_counts[student_counts < self.schema.min_sequence_length]
        
        if len(short_sequences) > 0:
            issues.append(
                f"Found {len(short_sequences)} students with less than "
                f"{self.schema.min_sequence_length} time points"
            )
        
        return issues


class FeatureExtractor:
    """Extracts features based on schema configuration."""
    
    def __init__(self, schema: DataSchema):
        self.schema = schema
        self.feature_indices = schema.get_feature_indices()
    
    def extract_features(self, row: pd.Series) -> List[float]:
        """Extract features from a data row based on schema."""
        features = []
        
        for col in self.schema.feature_columns:
            if col in row:
                value = self._safe_float_conversion(row[col])
            else:
                # Use default value from schema
                col_schema = self.schema.columns.get(col)
                if col_schema and col_schema.default_value is not None:
                    value = float(col_schema.default_value)
                else:
                    value = 0.0
            
            features.append(value)
        
        return features
    
    def extract_from_array(self, X: np.ndarray, feature_name: str) -> np.ndarray:
        """Extract specific feature from array using schema-based indexing."""
        if feature_name not in self.feature_indices:
            raise ValueError(f"Feature '{feature_name}' not in schema")
        
        idx = self.feature_indices[feature_name]
        return X[..., idx]
    
    def _safe_float_conversion(self, value: Any) -> float:
        """Safely convert value to float with consistent handling."""
        if pd.isna(value) or value == 'NA' or value == '':
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0


def week_string_to_numeric(week_str: Any, format: Optional[str] = None) -> float:
    """
    Convert week string to numeric with consistent float return type.
    
    Args:
        week_str: Week string or numeric value
        format: Optional format string for parsing
        
    Returns:
        float: Always returns float for consistency
    """
    if pd.isna(week_str) or week_str == 'NA' or week_str == '':
        return 0.0
    
    try:
        # Handle week string format (e.g., '2011-W36')
        if isinstance(week_str, str) and '-W' in week_str:
            year, week = week_str.split('-W')
            return float(int(year) * 100 + int(week))
        else:
            # Direct numeric conversion
            return float(week_str)
    except (ValueError, AttributeError, TypeError):
        return 0.0


def get_schema(name: str) -> DataSchema:
    """
    Get a predefined schema by name.
    
    Available schemas:
    - legacy: Original framework columns (name/week/proficient)
    - student_week: Student weekly aggregation columns
    - extended: Student weekly with ability features
    - time_goal: Predict weekly time spent (basic features)
    - time_goal_extended: Predict weekly time spent (with ability features)
    """
    if name not in SCHEMAS:
        raise ValueError(f"Unknown schema: {name}. Available: {list(SCHEMAS.keys())}")
    
    return SCHEMAS[name]


# Predefined schemas for common use cases
SCHEMAS = {
    'legacy': DataSchema(
        student_column='name',
        time_column='week',
        target_column='proficient',
        feature_columns=['week', 'proficient'],
        time_format='numeric',
        validation_rules={
            'proficient': {'min': 0.0, 'max': 1.0}
        }
    ),
    
    'student_week': DataSchema(
        student_column='anon_student_id',
        time_column='week_id',
        target_column='avg_proficiency',
        feature_columns=[
            'week_id', 'minutes_per_week', 'problems_solved',
            'total_opportunities', 'avg_proficiency', 'n_skills_measured'
        ],
        time_format='week_string',
        validation_rules={
            'avg_proficiency': {'min': 0.0, 'max': 1.0},
            'minutes_per_week': {'min': 0.0}
        }
    ),
    
    'extended': DataSchema(
        student_column='anon_student_id',
        time_column='week_id',
        target_column='avg_proficiency',
        feature_columns=[
            'week_id', 'minutes_per_week', 'problems_solved',
            'total_opportunities', 'avg_proficiency', 'n_skills_measured',
            'week_difficulty', 'student_ability', 'student_learning_rate'
        ],
        time_format='week_string',
        validation_rules={
            'avg_proficiency': {'min': 0.0, 'max': 1.0},
            'minutes_per_week': {'min': 0.0},
            'student_ability': {'min': 0.0, 'max': 1.0},
            'week_difficulty': {'min': 0.0, 'max': 1.0}
        }
    ),
    
    # New schemas for predicting time spent
    'time_goal': DataSchema(
        student_column='anon_student_id',
        time_column='week_id',
        target_column='minutes_per_week',  # Predicting engagement!
        feature_columns=[
            'week_id', 'avg_proficiency', 'problems_solved',
            'total_opportunities', 'n_skills_measured', 'minutes_per_week'
        ],
        time_format='week_string',
        validation_rules={
            'minutes_per_week': {'min': 0.0},
            'avg_proficiency': {'min': 0.0, 'max': 1.0}
        }
    ),
    
    'time_goal_extended': DataSchema(
        student_column='anon_student_id',
        time_column='week_id',
        target_column='minutes_per_week',  # Predicting engagement!
        feature_columns=[
            'week_id', 'avg_proficiency', 'problems_solved',
            'total_opportunities', 'n_skills_measured', 'week_difficulty',
            'student_ability', 'student_learning_rate', 'minutes_per_week'
        ],
        time_format='week_string',
        validation_rules={
            'minutes_per_week': {'min': 0.0},
            'avg_proficiency': {'min': 0.0, 'max': 1.0},
            'student_ability': {'min': 0.0, 'max': 1.0},
            'week_difficulty': {'min': 0.0, 'max': 1.0}
        }
    ),
    
    # Schema variants with different student ID strategies
    'time_goal_extended_target_encoding': DataSchema(
        student_column='anon_student_id',
        time_column='week_id',
        target_column='minutes_per_week',
        feature_columns=[
            'week_id', 'avg_proficiency', 'problems_solved',
            'total_opportunities', 'n_skills_measured', 'week_difficulty',
            'student_ability', 'student_learning_rate', 'minutes_per_week',
            'student_target_mean', 'student_target_std', 'student_target_count',
            'student_target_median', 'student_target_deviation'
        ],
        student_id_strategy=StudentIDStrategy('target_encoding'),
        time_format='week_string',
        validation_rules={
            'minutes_per_week': {'min': 0.0},
            'avg_proficiency': {'min': 0.0, 'max': 1.0},
            'student_ability': {'min': 0.0, 'max': 1.0},
            'week_difficulty': {'min': 0.0, 'max': 1.0}
        }
    ),
    
    'time_goal_extended_embeddings': DataSchema(
        student_column='anon_student_id',
        time_column='week_id',
        target_column='minutes_per_week',
        feature_columns=[
            'week_id', 'avg_proficiency', 'problems_solved',
            'total_opportunities', 'n_skills_measured', 'week_difficulty',
            'student_ability', 'student_learning_rate', 'minutes_per_week',
            'student_id_numeric'
        ],
        student_id_strategy=StudentIDStrategy('embeddings'),
        time_format='week_string',
        validation_rules={
            'minutes_per_week': {'min': 0.0},
            'avg_proficiency': {'min': 0.0, 'max': 1.0},
            'student_ability': {'min': 0.0, 'max': 1.0},
            'week_difficulty': {'min': 0.0, 'max': 1.0}
        }
    ),
    
    # Universal schema that works with ALL model types including mixed effects
    'time_goal_extended_universal': DataSchema(
        student_column='anon_student_id',
        time_column='week_id',
        target_column='minutes_per_week',
        feature_columns=[
            'anon_student_id',  # FIRST for mixed effects compatibility
            'week_id', 'avg_proficiency', 'problems_solved',
            'total_opportunities', 'n_skills_measured', 'week_difficulty',
            'student_ability', 'student_learning_rate', 'minutes_per_week'
        ],
        student_id_strategy=StudentIDStrategy('universal'),
        time_format='week_string',
        validation_rules={
            'minutes_per_week': {'min': 0.0},
            'avg_proficiency': {'min': 0.0, 'max': 1.0},
            'student_ability': {'min': 0.0, 'max': 1.0},
            'week_difficulty': {'min': 0.0, 'max': 1.0}
        }
    )
} 