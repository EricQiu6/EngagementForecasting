# Time Series Prediction Framework

A schema-driven framework for time series prediction supporting both traditional machine learning and deep learning models.

## Framework Structure

```
src/framework/
├── core/                   # Core framework components
│   ├── base.py            # Abstract base classes
│   ├── data.py            # Schema-based dataset implementations
│   └── schema.py          # Data schema definitions and validation
├── adapters/              # Model adapters for different frameworks
│   ├── sklearn_adapter.py # SKLearn model integration
│   └── pytorch_adapter.py # PyTorch model integration
├── models/                # Pre-built model implementations
│   ├── baselines.py       # Simple baseline models
│   ├── neural_nets.py     # Neural network architectures
│   ├── student_ability_model.py  # Don't worry about this
│   └── zero_inflated_model.py    # Don't worry about this
├── utils/                 # Utility functions
│   └── device.py          # GPU/CPU device management
└── config.py              # Configuration management, haven't really tested this
```

## Core Idea

### 1. **Schema**

The framework uses data schemas to eliminate hardcoded column names and ambiguous indexing in tensors:

```python
from src.framework.core.schema import DataSchema, get_schema

# Use predefined schemas
schema = get_schema('student_week')  # For student weekly aggregation data
schema = get_schema('time_goal')     # For time goal prediction

# Or create custom schemas
custom_schema = DataSchema(
    student_column='user_id',
    time_column='timestamp',
    target_column='performance',
    feature_columns=['activity', 'duration', 'difficulty']
)
```

### 2. **Unified Model Interface**

All models (sklearn, PyTorch, custom) implement the same interface:

```python
# All models support the same methods
model.fit(train_data, val_data)
predictions = model.predict(test_data)
params = model.get_params()
model.save('model.pkl')
```

### Adding New Models

#### 1. **Adding a New Baseline Model**

Create your model in `src/framework/models/baselines.py`:

```python
class MyNewBaseline:
    """Description of your baseline model."""

    def __init__(self, param1=None):
        self.param1 = param1
        self.fitted = False
        self.metadata = None  # For feature metadata

    def set_feature_metadata(self, metadata):
        """Receive feature metadata from adapter (optional)."""
        self.metadata = metadata

    def fit(self, X, y):
        """Train the model."""
        # Your training logic here
        self.fitted = True
        return self

    def predict(self, X):
        """Make predictions."""
        if not self.fitted:
            raise ValueError("Must fit before predicting")
        # Your prediction logic here
        return predictions

    def get_params(self, deep=True):
        """For sklearn compatibility."""
        return {'param1': self.param1}
```

Then add it to your evaluation:

```python
# In comprehensive_evaluation.py
algorithms['my_baseline'] = {
    'model': MyNewBaseline(param1=value),
    'description': 'My new baseline model',
    'category': 'baseline'
}
```

#### 2. **Adding a New Neural Network Model**

Create your model in `src/framework/models/neural_nets.py`:

```python
class MyNeuralModel(nn.Module):
    """Description of your neural model."""

    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # Handle sequence input: (batch_size, seq_len, features)
        if len(x.shape) == 3:
            # Use last timestep or apply pooling
            x = x[:, -1, :]  # Use last timestep
        return self.layers(x)
```

Add it to the model factory:

```python
# In neural_nets.py create_model function
elif model_type == 'my_neural':
    return MyNeuralModel(**kwargs)
```

Use with PyTorchAdapter:

```python
from src.framework.adapters import PyTorchAdapter
from src.framework.models.neural_nets import MyNeuralModel

model = PyTorchAdapter(
    MyNeuralModel(input_size=len(schema.feature_columns)),
    schema=schema
)
```

#### 3. **Adding Domain-Specific Models**

Create a new file `src/framework/models/my_domain_model.py`:

```python
class MyDomainModel(nn.Module):
    """Model specific to my domain/use case."""

    def __init__(self, schema: Optional['DataSchema'] = None):
        super().__init__()
        self.schema = schema

        # Build feature indices from schema
        if schema:
            self.feature_indices = schema.get_feature_indices()
            # Validate required features
            required = ['feature1', 'feature2']
            missing = [f for f in required if f not in self.feature_indices]
            if missing:
                raise ValueError(f"Missing features: {missing}")

    def _get_feature_index(self, feature_name: str, fallback: int) -> int:
        """Get feature index from schema or use fallback."""
        if self.feature_indices:
            return self.feature_indices[feature_name]
        return fallback

    def forward(self, x):
        # Extract features using schema
        feature1_idx = self._get_feature_index('feature1', 0)
        feature2_idx = self._get_feature_index('feature2', 1)

        feature1 = x[:, -1, feature1_idx]  # Latest value
        feature2 = x[:, :, feature2_idx]   # Full sequence

        # Your model logic here
        return predictions
```

### Adding New Data Schemas! Important if more data added to csv

Add new schemas to `src/framework/core/schema.py`:

```python
# In the SCHEMAS dictionary
SCHEMAS = {
    # ... existing schemas ...

    'my_new_schema': DataSchema(
        student_column='participant_id',
        time_column='session_number',
        target_column='outcome_score',
        feature_columns=[
            'session_number',
            'baseline_measure',
            'intervention_type',
            'outcome_score'
        ],
        time_format='numeric',  # or 'week_string' for '2011-W36' format
        validation_rules={ # don't worry about this haven't done testing
            'outcome_score': {'min': 0.0, 'max': 100.0},
            'baseline_measure': {'min': 0.0}
        }
    ),
}
```

Use your new schema:

```python
schema = get_schema('my_new_schema')
dataset = SchemaBasedTimeSeriesDataset(
    data_path='my_data.csv',
# use data-analysis/student_week_aggregations_rolling_new.csv
    schema=schema
)
```

### Adding New Feature Engineering!!! Important right here

Extend the feature engineering in `src/framework/adapters/sklearn_adapter.py`. Note that feature extraction uses the schema system:

```python
# In the _create_all_features method, add new feature types:

def _create_all_features(self, X_all: np.ndarray) -> np.ndarray:
    # ... existing feature extraction ...

    # 7. Add your new feature type
    my_new_features = []

    if 'my_special_column' in self.feature_map:
        idx = self.feature_map['my_special_column']
        values = X_all[:, :, idx]

        # Create your features
        special_feature = np.some_function(values)
        my_new_features.append(special_feature)

    if my_new_features:
        new_feature_array = np.column_stack(my_new_features)
        feature_list.append(new_feature_array)

    # ... rest of method ...
```

Also update `get_feature_names()` to include your new features:

```python
def get_feature_names(self):
    # ... existing feature names ...

    # Add names for your new features
    if 'my_special_column' in self.feature_map:
        feature_names.append('my_special_feature')
```

### Adding New Adapters

Create a new adapter in `src/framework/adapters/`:

```python
# my_framework_adapter.py
from ..core.base import TimeSeriesModel

class MyFrameworkAdapter(TimeSeriesModel):
    """Adapter for integrating MyFramework models."""

    def __init__(self, my_model, schema=None):
        self.my_model = my_model
        self.schema = schema
        self.is_fitted = False

    def fit(self, train_data, val_data=None, **kwargs):
        # Convert data format if needed
        X_train, y_train = self._convert_data(train_data)

        # Train using your framework's API
        self.my_model.train(X_train, y_train)
        self.is_fitted = True

        return {'status': 'completed'}

    def predict(self, data):
        X = self._convert_data(data, prediction=True)
        return self.my_model.predict(X)

    def _convert_data(self, data, prediction=False):
        # Convert from DataLoader or arrays to your format
        pass
```

Add to both `adapters/__init__.py` and `core/__init__.py`:

```python
# In adapters/__init__.py
from .my_framework_adapter import MyFrameworkAdapter

__all__ = [..., 'MyFrameworkAdapter']

# In core/__init__.py (if exposing at framework level)
from ..adapters import MyFrameworkAdapter
```

## Running Evaluations

### Basic Usage

```python
# Run evaluation with specific schema
python comprehensive_evaluation.py --schema time_goal

# Available schemas: legacy, student_week, extended, time_goal, time_goal_extended

# USE time_goal_extended mostly because it matches with the csv used under
# data-analysis/student_week_aggregations_rolling_new.csv
```

### Custom Evaluation

```python
from src.framework.core.schema import get_schema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from src.framework.adapters import SKLearnAdapter
from sklearn.ensemble import RandomForestRegressor

# Setup
schema = get_schema('time_goal_extended')
dataset = SchemaBasedTimeSeriesDataset('../data-analysis/student_week_aggregations_rolling_new.csv', schema)

# Create model
model = SKLearnAdapter(
    RandomForestRegressor(n_estimators=100),
    schema=schema
)

# Evaluate
cv = CrossValidator(model, dataset)
results = cv.cross_validate(n_splits=5)
```

## Example: Complete Custom Model

adding a new domain-specific model:

```python
# 1. Define schema
schema = DataSchema(
    student_column='user_id',
    time_column='week',
    target_column='engagement_score',
    feature_columns=['week', 'activity_count', 'engagement_score', 'difficulty']
)

# 2. Create model
class EngagementPredictor(nn.Module):
    def __init__(self, schema):
        super().__init__()
        self.schema = schema
        self.feature_indices = schema.get_feature_indices()

        # Simple neural network
        self.layers = nn.Sequential(
            nn.Linear(len(schema.feature_columns) * 5, 64),  # 5 timesteps
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        x_flat = x.view(batch_size, -1)  # Flatten
        return self.layers(x_flat)

# 3. Use with adapter
model = PyTorchAdapter(EngagementPredictor(schema), schema=schema)

# 4. Evaluate
dataset = SchemaBasedTimeSeriesDataset('data.csv', schema)
cv = CrossValidator(model, dataset)
results = cv.cross_validate()
```
