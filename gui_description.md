# Life Expectancy Forecasting - GUI Interface

The desktop GUI application provides an intuitive interface for exploring and predicting life expectancy data. Below is a description of each tab in the interface:

## Tab 1: Data Loading

![Data Loading Tab](data_loading_tab_placeholder.png)

This tab allows users to:
- Browse and select CSV dataset files
- Configure data loading options (separator, decimal format)
- Preview the dataset before proceeding
- Load the data into the application

## Tab 2: Data Exploration

![Data Exploration Tab](data_exploration_tab_placeholder.png)

This tab enables:
- Selection of different visualization types:
  - Life Expectancy Trends
  - Regional Differences
  - Gender Gap Analysis
- Country selection for trend comparison
- Interactive visualization of data patterns

## Tab 3: Model Training

![Model Training Tab](model_training_tab_placeholder.png)

This tab provides:
- Configuration of model parameters:
  - Sequence length (years of history to use)
  - Forecast horizon (years to predict)
  - Training epochs
- Model training with progress tracking
- Visual learning curves
- Performance metrics display
- Option to load existing models

## Tab 4: Predictions

![Predictions Tab](predictions_tab_placeholder.png)

This tab offers:
- Selection of countries for prediction
- Configuration of prediction parameters
- Visualization of historical data and forecasts
- Tabular display of prediction results
- Export functionality to save predictions to CSV

## Usage Flow

1. Start by loading your dataset in the **Data Loading** tab
2. Explore patterns in the data using the **Data Exploration** tab
3. Train a model with your chosen parameters in the **Model Training** tab
4. Generate and visualize predictions in the **Predictions** tab
5. Export your results for further analysis or reporting

The application handles all the complex transformer model operations behind the scenes, presenting a user-friendly interface that requires no coding knowledge to use. 