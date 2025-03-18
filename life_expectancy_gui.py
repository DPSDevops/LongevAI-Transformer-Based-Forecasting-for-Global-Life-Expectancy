import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import torch

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QTabWidget, QFileDialog, QProgressBar, QSpinBox,
                             QCheckBox, QGridLayout, QGroupBox, QTableWidget,
                             QTableWidgetItem, QMessageBox, QSplitter, QFrame,
                             QListWidget, QListWidgetItem, QLineEdit, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPixmap

# Import the life expectancy forecasting model functions
from life_expectancy_analysis import (load_data, explore_data, prepare_data, 
                                     TimeSeriesTransformer, train_model, 
                                     make_predictions, predict_future_life_expectancy)

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for displaying plots in the GUI"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)

class TrainingThread(QThread):
    """Thread for training the model without freezing the GUI"""
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    
    def __init__(self, df, seq_length, forecast_horizon, epochs):
        super().__init__()
        self.df = df
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.epochs = epochs
        
    def run(self):
        # Prepare data for modeling
        train_loader, test_loader, scaler, encoder, feature_dim = prepare_data(
            self.df, self.seq_length, self.forecast_horizon)
        
        # Get input and output dimensions
        for x, y in train_loader:
            input_dim = feature_dim
            output_dim = y.shape[1]
            break
        
        # Create a custom train function that emits progress
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TimeSeriesTransformer(input_dim, output_dim).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        train_losses = []
        test_losses = []
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Evaluation
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            
            # Emit progress signal (0-100%)
            progress = int((epoch + 1) / self.epochs * 100)
            self.progress_signal.emit(progress)
            
        # Save the model
        torch.save(model.state_dict(), 'life_expectancy_transformer.pth')
        
        # Make predictions on test set
        predictions, targets = make_predictions(model, test_loader, device)
        mse = np.mean((predictions - targets) ** 2)
        
        # Return the trained model and metrics
        result = {
            'model': model,
            'scaler': scaler,
            'encoder': encoder,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_mse': mse
        }
        self.finished_signal.emit(result)

class LifeExpectancyGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Life Expectancy Forecasting Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize data and model variables
        self.df = None
        self.model = None
        self.scaler = None
        self.encoder = None
        
        # Set up the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tabs for different functionalities
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs for each major function
        self.data_tab = QWidget()
        self.eda_tab = QWidget()
        self.model_tab = QWidget()
        self.prediction_tab = QWidget()
        
        self.tabs.addTab(self.data_tab, "Data Loading")
        self.tabs.addTab(self.eda_tab, "Data Exploration")
        self.tabs.addTab(self.model_tab, "Model Training")
        self.tabs.addTab(self.prediction_tab, "Predictions")
        
        # Set up individual tabs
        self.setup_data_tab()
        self.setup_eda_tab()
        self.setup_model_tab()
        self.setup_prediction_tab()
        
        # Status bar for messages
        self.statusBar().showMessage("Ready")
    
    def setup_data_tab(self):
        """Set up the data loading tab"""
        layout = QVBoxLayout(self.data_tab)
        
        # File selection group
        file_group = QGroupBox("Data Loading")
        file_layout = QVBoxLayout()
        
        # File selection widgets
        self.file_path_label = QLabel("No file selected")
        self.file_browse_button = QPushButton("Browse")
        self.file_browse_button.clicked.connect(self.browse_file)
        
        file_layout.addWidget(QLabel("Select CSV file with life expectancy data:"))
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(self.file_browse_button)
        
        # Data loading options
        options_group = QGroupBox("Data Loading Options")
        options_layout = QGridLayout()
        
        self.separator_combo = QComboBox()
        self.separator_combo.addItems([";", ",", "tab", "|", "space"])
        
        self.decimal_combo = QComboBox()
        self.decimal_combo.addItems([",", "."])
        
        options_layout.addWidget(QLabel("Separator:"), 0, 0)
        options_layout.addWidget(self.separator_combo, 0, 1)
        options_layout.addWidget(QLabel("Decimal:"), 1, 0)
        options_layout.addWidget(self.decimal_combo, 1, 1)
        
        options_group.setLayout(options_layout)
        
        # Load button
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data_file)
        
        # Data preview
        data_preview_group = QGroupBox("Data Preview")
        data_preview_layout = QVBoxLayout()
        
        self.data_preview_table = QTableWidget()
        data_preview_layout.addWidget(self.data_preview_table)
        data_preview_group.setLayout(data_preview_layout)
        
        # Add everything to main layout
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        layout.addWidget(options_group)
        layout.addWidget(self.load_button)
        layout.addWidget(data_preview_group)
    
    def setup_eda_tab(self):
        """Set up the exploratory data analysis tab"""
        layout = QVBoxLayout(self.eda_tab)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Visualization options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QGridLayout()
        
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Life Expectancy Trends", 
            "Regional Differences", 
            "Gender Gap Analysis"
        ])
        
        # Country selection for trends
        self.country_list = QListWidget()
        self.country_list.setSelectionMode(QListWidget.MultiSelection)
        
        viz_layout.addWidget(QLabel("Visualization Type:"), 0, 0)
        viz_layout.addWidget(self.viz_type_combo, 0, 1)
        viz_layout.addWidget(QLabel("Select Countries:"), 1, 0)
        viz_layout.addWidget(self.country_list, 1, 1)
        
        viz_group.setLayout(viz_layout)
        controls_layout.addWidget(viz_group)
        
        # Generate visualization button
        self.generate_viz_button = QPushButton("Generate Visualization")
        self.generate_viz_button.clicked.connect(self.generate_visualization)
        controls_layout.addWidget(self.generate_viz_button)
        
        # Matplotlib canvas for plotting
        plot_group = QGroupBox("Visualization")
        plot_layout = QVBoxLayout()
        
        self.eda_canvas = MatplotlibCanvas(self.eda_tab, width=10, height=6)
        plot_layout.addWidget(self.eda_canvas)
        
        plot_group.setLayout(plot_layout)
        
        # Add to main layout
        layout.addLayout(controls_layout)
        layout.addWidget(plot_group)
    
    def setup_model_tab(self):
        """Set up the model training tab"""
        layout = QVBoxLayout(self.model_tab)
        
        # Model parameters group
        params_group = QGroupBox("Model Parameters")
        params_layout = QGridLayout()
        
        self.seq_length_spin = QSpinBox()
        self.seq_length_spin.setRange(1, 20)
        self.seq_length_spin.setValue(10)
        
        self.forecast_horizon_spin = QSpinBox()
        self.forecast_horizon_spin.setRange(1, 10)
        self.forecast_horizon_spin.setValue(5)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(50)
        
        params_layout.addWidget(QLabel("Sequence Length:"), 0, 0)
        params_layout.addWidget(self.seq_length_spin, 0, 1)
        params_layout.addWidget(QLabel("Forecast Horizon:"), 1, 0)
        params_layout.addWidget(self.forecast_horizon_spin, 1, 1)
        params_layout.addWidget(QLabel("Training Epochs:"), 2, 0)
        params_layout.addWidget(self.epochs_spin, 2, 1)
        
        params_group.setLayout(params_layout)
        
        # Training controls
        train_controls_layout = QHBoxLayout()
        
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        
        self.load_model_button = QPushButton("Load Existing Model")
        self.load_model_button.clicked.connect(self.load_existing_model)
        
        train_controls_layout.addWidget(self.train_button)
        train_controls_layout.addWidget(self.load_model_button)
        
        # Training progress
        self.train_progress = QProgressBar()
        
        # Results display
        results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout()
        
        self.training_results_canvas = MatplotlibCanvas(self.model_tab, width=10, height=6)
        results_layout.addWidget(self.training_results_canvas)
        
        # Training metrics
        metrics_layout = QHBoxLayout()
        self.train_loss_label = QLabel("Training Loss: N/A")
        self.test_loss_label = QLabel("Test Loss: N/A")
        self.mse_label = QLabel("Test MSE: N/A")
        
        metrics_layout.addWidget(self.train_loss_label)
        metrics_layout.addWidget(self.test_loss_label)
        metrics_layout.addWidget(self.mse_label)
        
        results_layout.addLayout(metrics_layout)
        results_group.setLayout(results_layout)
        
        # Add to main layout
        layout.addWidget(params_group)
        layout.addLayout(train_controls_layout)
        layout.addWidget(self.train_progress)
        layout.addWidget(results_group)
    
    def setup_prediction_tab(self):
        """Set up the prediction tab"""
        layout = QVBoxLayout(self.prediction_tab)
        
        # Control panel
        controls_layout = QHBoxLayout()
        
        # Country selection for prediction
        country_group = QGroupBox("Select Countries")
        country_layout = QVBoxLayout()
        
        self.prediction_country_list = QListWidget()
        self.prediction_country_list.setSelectionMode(QListWidget.MultiSelection)
        
        country_layout.addWidget(self.prediction_country_list)
        country_group.setLayout(country_layout)
        
        # Prediction parameters
        params_group = QGroupBox("Prediction Parameters")
        params_layout = QGridLayout()
        
        self.prediction_years_spin = QSpinBox()
        self.prediction_years_spin.setRange(1, 10)
        self.prediction_years_spin.setValue(5)
        
        params_layout.addWidget(QLabel("Years to Predict:"), 0, 0)
        params_layout.addWidget(self.prediction_years_spin, 0, 1)
        
        # Predict button
        self.predict_button = QPushButton("Generate Predictions")
        self.predict_button.clicked.connect(self.generate_predictions)
        
        params_layout.addWidget(self.predict_button, 1, 0, 1, 2)
        params_group.setLayout(params_layout)
        
        controls_layout.addWidget(country_group)
        controls_layout.addWidget(params_group)
        
        # Results visualization
        results_splitter = QSplitter(Qt.Horizontal)
        
        # Visualization panel
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        
        self.prediction_canvas = MatplotlibCanvas(self.prediction_tab, width=8, height=6)
        viz_layout.addWidget(self.prediction_canvas)
        
        # Results table panel
        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        
        table_layout.addWidget(QLabel("Prediction Results"))
        self.prediction_table = QTableWidget()
        table_layout.addWidget(self.prediction_table)
        
        # Country selector for shown prediction
        self.prediction_country_selector = QComboBox()
        self.prediction_country_selector.currentIndexChanged.connect(self.update_prediction_display)
        table_layout.addWidget(QLabel("Select Country to Display:"))
        table_layout.addWidget(self.prediction_country_selector)
        
        # Export button
        self.export_button = QPushButton("Export Predictions to CSV")
        self.export_button.clicked.connect(self.export_predictions)
        table_layout.addWidget(self.export_button)
        
        results_splitter.addWidget(viz_panel)
        results_splitter.addWidget(table_panel)
        
        # Add to main layout
        layout.addLayout(controls_layout)
        layout.addWidget(results_splitter)

    def browse_file(self):
        """Open file dialog to select a CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)")
        
        if file_path:
            self.file_path_label.setText(file_path)
    
    def load_data_file(self):
        """Load the selected CSV file and display a preview"""
        file_path = self.file_path_label.text()
        if file_path == "No file selected":
            QMessageBox.warning(self, "No File Selected", 
                               "Please select a CSV file first.")
            return
        
        # Get separator and decimal options
        separator = self.separator_combo.currentText()
        if separator == "tab":
            separator = "\t"
        elif separator == "space":
            separator = " "
            
        decimal = self.decimal_combo.currentText()
        
        try:
            # Load data
            self.df = pd.read_csv(file_path, sep=separator, decimal=decimal)
            
            # Update preview table
            self.update_data_preview()
            
            # Update country lists
            self.update_country_lists()
            
            self.statusBar().showMessage(f"Loaded data with {len(self.df)} rows and {len(self.df.columns)} columns")
            
            # Enable other tabs
            self.tabs.setTabEnabled(1, True)  # EDA tab
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Data", 
                                f"Failed to load data: {str(e)}")
    
    def update_data_preview(self):
        """Update the data preview table with loaded data"""
        if self.df is None:
            return
        
        # Show first 10 rows
        preview_df = self.df.head(10)
        
        self.data_preview_table.setRowCount(len(preview_df))
        self.data_preview_table.setColumnCount(len(preview_df.columns))
        self.data_preview_table.setHorizontalHeaderLabels(preview_df.columns)
        
        # Fill table with data
        for i, row in enumerate(preview_df.itertuples()):
            for j, value in enumerate(row[1:]):
                self.data_preview_table.setItem(i, j, QTableWidgetItem(str(value)))
        
        self.data_preview_table.resizeColumnsToContents()
    
    def update_country_lists(self):
        """Update the country selection lists in EDA and Prediction tabs"""
        if self.df is None:
            return
        
        # Clear existing items
        self.country_list.clear()
        self.prediction_country_list.clear()
        
        # Get unique countries
        if 'country_code' in self.df.columns:
            countries = self.df['country_code'].unique()
            
            # Add to lists
            for country in countries:
                country_name = country
                # Try to get full name if available
                if 'country_name' in self.df.columns:
                    name_row = self.df[self.df['country_code'] == country].iloc[0]
                    if 'country_name' in name_row:
                        country_name = f"{country} - {name_row['country_name']}"
                
                self.country_list.addItem(country)
                self.prediction_country_list.addItem(country)
    
    def generate_visualization(self):
        """Generate the selected visualization on the EDA tab"""
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        viz_type = self.viz_type_combo.currentText()
        
        # Clear the canvas
        self.eda_canvas.axes.clear()
        
        if viz_type == "Life Expectancy Trends":
            # Get selected countries
            selected_countries = [item.text() for item in self.country_list.selectedItems()]
            
            if not selected_countries:
                QMessageBox.warning(self, "No Selection", 
                                  "Please select at least one country.")
                return
            
            for country in selected_countries:
                country_data = self.df[self.df['country_code'] == country]
                if not country_data.empty:
                    self.eda_canvas.axes.plot(
                        country_data['year'], country_data['life_expectancy_women'], 
                        label=f"{country} - Women")
                    self.eda_canvas.axes.plot(
                        country_data['year'], country_data['life_expectancy_men'], 
                        linestyle='--', label=f"{country} - Men")
            
            self.eda_canvas.axes.set_title('Life Expectancy Trends')
        
        elif viz_type == "Regional Differences":
            # Group data by region and year
            region_data = self.df.groupby(['region', 'year']).agg({
                'life_expectancy_women': 'mean',
                'life_expectancy_men': 'mean'
            }).reset_index()
            
            # Plot each region
            for region in region_data['region'].unique():
                region_subset = region_data[region_data['region'] == region]
                self.eda_canvas.axes.plot(
                    region_subset['year'], region_subset['life_expectancy_women'], 
                    label=f"{region} - Women")
            
            self.eda_canvas.axes.set_title('Regional Life Expectancy Trends - Women')
        
        elif viz_type == "Gender Gap Analysis":
            # Calculate gender gap
            gender_gap = self.df['life_expectancy_women'] - self.df['life_expectancy_men']
            self.df['gender_gap'] = gender_gap
            
            # Plot global trend
            yearly_gap = self.df.groupby('year')['gender_gap'].mean()
            self.eda_canvas.axes.plot(yearly_gap.index, yearly_gap.values)
            self.eda_canvas.axes.set_title('Global Gender Gap in Life Expectancy (Women - Men)')
        
        # Common settings
        self.eda_canvas.axes.set_xlabel('Year')
        self.eda_canvas.axes.set_ylabel('Life Expectancy (years)')
        self.eda_canvas.axes.legend()
        self.eda_canvas.axes.grid(True)
        
        # Redraw the canvas
        self.eda_canvas.draw()
    
    def train_model(self):
        """Start the model training process"""
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        # Get training parameters
        seq_length = self.seq_length_spin.value()
        forecast_horizon = self.forecast_horizon_spin.value()
        epochs = self.epochs_spin.value()
        
        # Create and start training thread
        self.training_thread = TrainingThread(
            self.df, seq_length, forecast_horizon, epochs)
        
        # Connect signals
        self.training_thread.progress_signal.connect(self.update_training_progress)
        self.training_thread.finished_signal.connect(self.training_finished)
        
        # Start training
        self.train_button.setEnabled(False)
        self.training_thread.start()
        
        self.statusBar().showMessage("Training in progress...")
    
    def update_training_progress(self, value):
        """Update the training progress bar"""
        self.train_progress.setValue(value)
    
    def training_finished(self, result):
        """Handle the completion of model training"""
        # Store model and related objects
        self.model = result['model']
        self.scaler = result['scaler']
        self.encoder = result['encoder']
        
        # Update UI
        self.train_button.setEnabled(True)
        self.statusBar().showMessage("Training completed!")
        
        # Update metrics
        self.train_loss_label.setText(f"Training Loss: {result['train_losses'][-1]:.4f}")
        self.test_loss_label.setText(f"Test Loss: {result['test_losses'][-1]:.4f}")
        self.mse_label.setText(f"Test MSE: {result['test_mse']:.4f}")
        
        # Plot learning curves
        self.training_results_canvas.axes.clear()
        self.training_results_canvas.axes.plot(result['train_losses'], label='Training Loss')
        self.training_results_canvas.axes.plot(result['test_losses'], label='Validation Loss')
        self.training_results_canvas.axes.set_title('Learning Curves')
        self.training_results_canvas.axes.set_xlabel('Epoch')
        self.training_results_canvas.axes.set_ylabel('MSE Loss')
        self.training_results_canvas.axes.legend()
        self.training_results_canvas.axes.grid(True)
        self.training_results_canvas.draw()
        
        # Enable prediction tab
        self.tabs.setTabEnabled(3, True)  # Prediction tab
    
    def load_existing_model(self):
        """Load a previously trained model"""
        try:
            # Check if model file exists
            if not os.path.exists('life_expectancy_transformer.pth'):
                QMessageBox.warning(self, "Model Not Found", 
                                  "No saved model found. Please train a model first.")
                return
            
            # Ask user to select model file
            model_path, _ = QFileDialog.getOpenFileName(
                self, "Select Model File", "", "PyTorch Model (*.pth);;All Files (*)")
            
            if not model_path:
                return
            
            QMessageBox.information(self, "Model Loading", 
                                  "Model loading is partially implemented. You'll need to train a new model "
                                  "to get the proper scaler and encoder objects.")
            
            # Enable prediction tab
            self.tabs.setTabEnabled(3, True)  # Prediction tab
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", 
                                f"Failed to load model: {str(e)}")
    
    def generate_predictions(self):
        """Generate predictions for selected countries"""
        if self.model is None or self.scaler is None or self.encoder is None:
            QMessageBox.warning(self, "No Model", 
                              "Please train a model first.")
            return
        
        # Get selected countries
        selected_countries = [item.text() for item in self.prediction_country_list.selectedItems()]
        
        if not selected_countries:
            QMessageBox.warning(self, "No Selection", 
                              "Please select at least one country.")
            return
        
        # Get prediction parameters
        forecast_years = self.prediction_years_spin.value()
        
        try:
            # Generate predictions
            self.predictions = predict_future_life_expectancy(
                self.model, self.df, selected_countries, 
                self.scaler, self.encoder, forecast_years)
            
            # Update country selector for showing predictions
            self.prediction_country_selector.clear()
            for country in self.predictions.keys():
                self.prediction_country_selector.addItem(country)
            
            # Show first country prediction
            if self.predictions:
                self.update_prediction_display()
            
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", 
                                f"Error generating predictions: {str(e)}")
    
    def update_prediction_display(self):
        """Update the prediction display for the selected country"""
        if not hasattr(self, 'predictions') or not self.predictions:
            return
        
        # Get selected country
        selected_country = self.prediction_country_selector.currentText()
        if not selected_country or selected_country not in self.predictions:
            return
        
        # Get prediction data
        pred_data = self.predictions[selected_country]
        country_name = pred_data['country_name']
        future_years = pred_data['future_years']
        women_predictions = pred_data['women_predictions']
        men_predictions = pred_data['men_predictions']
        
        # Update table
        self.prediction_table.setRowCount(len(future_years))
        self.prediction_table.setColumnCount(3)
        self.prediction_table.setHorizontalHeaderLabels(['Year', 'Women', 'Men'])
        
        for i, year in enumerate(future_years):
            self.prediction_table.setItem(i, 0, QTableWidgetItem(str(year)))
            self.prediction_table.setItem(i, 1, QTableWidgetItem(f"{women_predictions[i]:.2f}"))
            self.prediction_table.setItem(i, 2, QTableWidgetItem(f"{men_predictions[i]:.2f}"))
        
        self.prediction_table.resizeColumnsToContents()
        
        # Update plot
        self.prediction_canvas.axes.clear()
        
        # Get historical data
        country_data = self.df[self.df['country_code'] == selected_country]
        
        # Plot historical data
        self.prediction_canvas.axes.plot(
            country_data['year'], country_data['life_expectancy_women'], 
            label='Historical (Women)', color='blue')
        self.prediction_canvas.axes.plot(
            country_data['year'], country_data['life_expectancy_men'], 
            label='Historical (Men)', color='green')
        
        # Plot predictions
        self.prediction_canvas.axes.plot(
            future_years, women_predictions, 'o--', 
            label='Predicted (Women)', color='blue')
        self.prediction_canvas.axes.plot(
            future_years, men_predictions, 'o--', 
            label='Predicted (Men)', color='green')
        
        self.prediction_canvas.axes.set_title(f'Life Expectancy Forecast for {country_name} ({selected_country})')
        self.prediction_canvas.axes.set_xlabel('Year')
        self.prediction_canvas.axes.set_ylabel('Life Expectancy (years)')
        self.prediction_canvas.axes.legend()
        self.prediction_canvas.axes.grid(True)
        
        self.prediction_canvas.draw()
    
    def export_predictions(self):
        """Export predictions to CSV file"""
        if not hasattr(self, 'predictions') or not self.predictions:
            QMessageBox.warning(self, "No Predictions", 
                              "Please generate predictions first.")
            return
        
        # Ask for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Predictions", "", "CSV Files (*.csv);;All Files (*)")
        
        if not file_path:
            return
        
        try:
            # Create DataFrame from predictions
            dfs = []
            
            for country_code, data in self.predictions.items():
                country_df = pd.DataFrame({
                    'country_code': country_code,
                    'country_name': data['country_name'],
                    'year': data['future_years'],
                    'life_expectancy_women': data['women_predictions'],
                    'life_expectancy_men': data['men_predictions']
                })
                dfs.append(country_df)
            
            # Combine all predictions
            all_predictions = pd.concat(dfs, ignore_index=True)
            
            # Save to CSV
            all_predictions.to_csv(file_path, index=False)
            
            QMessageBox.information(self, "Export Complete", 
                                 f"Predictions exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                                f"Error exporting predictions: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = LifeExpectancyGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 