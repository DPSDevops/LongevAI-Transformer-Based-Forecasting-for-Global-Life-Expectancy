import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load the dataset
def load_data():
    print("Loading dataset...")
    # Note: The dataset uses semicolons as separators and commas for decimal points
    df = pd.read_csv('life_expectancy_dataset.csv', sep=';', decimal=',')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

# Exploratory Data Analysis
def explore_data(df):
    print("\nExploratory Data Analysis:")
    
    # Basic statistics
    print("\nBasic Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Visualize life expectancy trends over time
    plt.figure(figsize=(12, 6))
    
    # Sample a few countries to visualize
    countries = ['USA', 'CHN', 'IND', 'GBR', 'BRA']
    for country in countries:
        country_data = df[df['country_code'] == country]
        if not country_data.empty:
            plt.plot(country_data['year'], country_data['life_expectancy_women'], 
                     label=f"{country} - Women")
            plt.plot(country_data['year'], country_data['life_expectancy_men'], 
                     linestyle='--', label=f"{country} - Men")
    
    plt.title('Life Expectancy Trends (1960-2022)')
    plt.xlabel('Year')
    plt.ylabel('Life Expectancy (years)')
    plt.legend()
    plt.grid(True)
    plt.savefig('life_expectancy_trends.png')
    
    # Analyze regional differences
    plt.figure(figsize=(14, 8))
    region_data = df.groupby(['region', 'year']).agg({
        'life_expectancy_women': 'mean',
        'life_expectancy_men': 'mean'
    }).reset_index()
    
    for region in region_data['region'].unique():
        region_subset = region_data[region_data['region'] == region]
        plt.plot(region_subset['year'], region_subset['life_expectancy_women'], 
                 label=f"{region} - Women")
    
    plt.title('Regional Life Expectancy Trends - Women')
    plt.xlabel('Year')
    plt.ylabel('Life Expectancy (years)')
    plt.legend()
    plt.grid(True)
    plt.savefig('regional_life_expectancy_women.png')
    
    # Gender gap analysis
    df['gender_gap'] = df['life_expectancy_women'] - df['life_expectancy_men']
    
    plt.figure(figsize=(12, 6))
    yearly_gap = df.groupby('year')['gender_gap'].mean()
    plt.plot(yearly_gap.index, yearly_gap.values)
    plt.title('Global Gender Gap in Life Expectancy (Women - Men)')
    plt.xlabel('Year')
    plt.ylabel('Gap (years)')
    plt.grid(True)
    plt.savefig('gender_gap_trend.png')

# Data preparation for the transformer model
class LifeExpectancyDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Get sequence of features and corresponding target
        x = self.features[idx]
        y = self.targets[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)

def prepare_data(df, seq_length=10, forecast_horizon=5):
    # Sort by country and year
    df = df.sort_values(['country_code', 'year'])
    
    # Convert categorical features using one-hot encoding
    categorical_features = ['region', 'sub-region']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[categorical_features])
    
    # Scale numerical features
    numerical_features = ['year', 'life_expectancy_women', 'life_expectancy_men']
    scaler = StandardScaler()
    scaled_nums = scaler.fit_transform(df[numerical_features])
    
    # Combine all features
    feature_matrix = np.hstack([scaled_nums, encoded_cats])
    
    # Get the feature dimension
    feature_dim = feature_matrix.shape[1]
    
    # Create sequences and targets
    sequences = []
    targets = []
    
    for country in df['country_code'].unique():
        country_data = df[df['country_code'] == country]
        country_features = feature_matrix[country_data.index]
        
        if len(country_data) < seq_length + forecast_horizon:
            continue
            
        for i in range(len(country_data) - seq_length - forecast_horizon + 1):
            seq = country_features[i:i+seq_length]
            # Target is the life expectancy for women and men for the next forecast_horizon years
            target = country_features[i+seq_length:i+seq_length+forecast_horizon, 1:3]  # Life expectancy columns
            sequences.append(seq)
            targets.append(target.flatten())  # Flatten the target for simplicity
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(sequences), np.array(targets), test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = LifeExpectancyDataset(X_train, y_train, seq_length)
    test_dataset = LifeExpectancyDataset(X_test, y_test, seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, scaler, encoder, feature_dim

# Transformer model for time series forecasting
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=8, num_layers=3, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        x = self.embedding(x)
        
        # No mask is used to allow the model to attend to all positions
        x = self.transformer_encoder(x)
        
        # Use only the last sequence element for prediction
        x = x[:, -1, :]
        
        x = self.fc_out(x)
        return x

# Training function
def train_model(train_loader, test_loader, input_dim, output_dim, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TimeSeriesTransformer(input_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
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
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves.png')
    
    return model

# Make predictions
def make_predictions(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets)

# Function to predict future life expectancy for specific countries
def predict_future_life_expectancy(model, df, country_codes, scaler, encoder, forecast_years=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Sort by country and year
    df = df.sort_values(['country_code', 'year'])
    
    # Get feature column information
    categorical_features = ['region', 'sub-region']
    numerical_features = ['year', 'life_expectancy_women', 'life_expectancy_men']
    
    # Initialize results dictionary
    results = {}
    
    for country_code in country_codes:
        # Get country data
        country_data = df[df['country_code'] == country_code].copy()
        
        if len(country_data) < 10:  # Need at least 10 years of history
            print(f"Insufficient data for {country_code}. Need at least 10 years of history.")
            continue
            
        # Sort by year
        country_data = country_data.sort_values('year')
        
        # Get the most recent 10 years of data
        recent_data = country_data.iloc[-10:]
        
        # Get the last year in the data
        last_year = recent_data['year'].max()
        
        # Prepare input features
        cat_data = encoder.transform(recent_data[categorical_features])
        num_data = scaler.transform(recent_data[numerical_features])
        
        # Combine features
        features = np.hstack([num_data, cat_data])
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            predictions = model(input_tensor)
            predictions = predictions.cpu().numpy()[0]
        
        # Reshape predictions to have women and men columns
        predictions = predictions.reshape(-1, 2)
        
        # Create future years
        future_years = np.arange(last_year + 1, last_year + forecast_years + 1)
        
        # Get country name
        country_name = country_data['country_name'].iloc[0]
        
        # Create DataFrames for visualization
        women_predictions = []
        men_predictions = []
        
        # Inverse transform the predictions to get actual life expectancy values
        for i, year in enumerate(future_years):
            # Create mock data for inverse transform
            mock_data = np.zeros((1, 3))
            mock_data[0, 0] = year
            mock_data[0, 1] = predictions[i, 0]  # Women prediction (scaled)
            mock_data[0, 2] = predictions[i, 1]  # Men prediction (scaled)
            
            # Inverse transform
            inverse_data = scaler.inverse_transform(mock_data)
            
            # Store the predicted life expectancy
            women_predictions.append(inverse_data[0, 1])
            men_predictions.append(inverse_data[0, 2])
        
        # Store results
        results[country_code] = {
            'country_name': country_name,
            'future_years': future_years,
            'women_predictions': women_predictions,
            'men_predictions': men_predictions
        }
        
        # Plot the historical and predicted data
        plt.figure(figsize=(12, 6))
        
        # Historical data
        plt.plot(country_data['year'], country_data['life_expectancy_women'], 
                 label='Historical (Women)', color='blue')
        plt.plot(country_data['year'], country_data['life_expectancy_men'], 
                 label='Historical (Men)', color='green')
        
        # Predicted data
        plt.plot(future_years, women_predictions, 'o--', 
                 label='Predicted (Women)', color='blue')
        plt.plot(future_years, men_predictions, 'o--', 
                 label='Predicted (Men)', color='green')
        
        plt.title(f'Life Expectancy Forecast for {country_name} ({country_code})')
        plt.xlabel('Year')
        plt.ylabel('Life Expectancy (years)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'forecast_{country_code}.png')
    
    return results

def main():
    # Load data
    df = load_data()
    
    # Explore data
    explore_data(df)
    
    # Prepare data for modeling
    seq_length = 10  # Use 10 years of history
    forecast_horizon = 5  # Predict 5 years ahead
    train_loader, test_loader, scaler, encoder, feature_dim = prepare_data(df, seq_length, forecast_horizon)
    
    # Get input and output dimensions
    for x, y in train_loader:
        input_dim = feature_dim
        output_dim = y.shape[1]  # Target dimension (women and men life expectancy for forecast_horizon years)
        break
    
    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_model(train_loader, test_loader, input_dim, output_dim, epochs=50)
    
    # Make predictions
    predictions, targets = make_predictions(model, test_loader, device)
    
    # Evaluate predictions
    mse = np.mean((predictions - targets) ** 2)
    print(f"Test MSE: {mse:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), 'life_expectancy_transformer.pth')
    print("Model saved successfully")
    
    # Predict future life expectancy for specific countries
    countries_to_predict = ['USA', 'CHN', 'IND', 'GBR', 'BRA']
    future_predictions = predict_future_life_expectancy(model, df, countries_to_predict, scaler, encoder)
    
    # Print predictions
    print("\nFuture Life Expectancy Predictions:")
    for country_code, data in future_predictions.items():
        print(f"\n{data['country_name']} ({country_code}):")
        print("Year | Women | Men")
        print("-" * 25)
        for i, year in enumerate(data['future_years']):
            print(f"{year} | {data['women_predictions'][i]:.2f} | {data['men_predictions'][i]:.2f}")

if __name__ == "__main__":
    main() 