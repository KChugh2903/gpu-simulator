import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from rocketpy import Environment, SolidMotor, Rocket, Flight
import requests
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import pytz
import argparse
import os
from datetime import datetime, timedelta

class AtmosphericNN(nn.Module):
    def __init__(self):
        super(AtmosphericNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),    # Input: altitude and latitude
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)     # Output: density, pressure, temperature, speed of sound, wind speed
        )
        
    def forward(self, x):
        return self.network(x)

class EnvironmentGenerator:
    def __init__(self, latitude, longitude, elevation):
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.max_altitude = 20000
        self.num_points = 2000
        self.model = AtmosphericNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tomorrow = datetime.now() + timedelta(days=1)
        self.env = Environment(
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            date=self.tomorrow
        )
        
    def generate_training_data(self):
        """Generate training data from RocketPy environment"""
        self.env.set_atmospheric_model(type="forecast", file="GFS")
        
        altitudes = np.linspace(0, self.max_altitude, self.num_points)
        latitudes = np.linspace(self.latitude - 1, self.latitude + 1, 10)
        
        X = [] 
        y = []  
        
        for lat in latitudes:
            for alt in altitudes:
                density = float(self.env.density(alt))
                pressure = float(self.env.pressure(alt))
                temperature = float(self.env.temperature(alt))
                speed_of_sound = float(self.env.speed_of_sound(alt))
                wind_speed = float(self.env.wind_speed(alt))
                
                X.append([alt, lat])
                y.append([density, pressure, temperature, speed_of_sound, wind_speed])
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def train_model(self, epochs=100, batch_size=32):
        """Train the neural network on atmospheric data"""
        X, y = self.generate_training_data()
        X = X.to(self.device)
        y = y.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())
        n_batches = len(X) // batch_size
        for epoch in range(epochs):
            total_loss = 0
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_X = X[start_idx:end_idx]
                batch_y = y[start_idx:end_idx]
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/n_batches:.4f}')
    
    def generate_environment_data(self):
        """Generate environment data using the trained model"""
        altitudes = np.linspace(0, self.max_altitude, self.num_points)
        data = []
        with torch.no_grad():
            for altitude in altitudes:
                input_data = torch.tensor([[altitude, self.latitude]], dtype=torch.float32).to(self.device)
                output = self.model(input_data)
                density, pressure, temperature, speed_of_sound, wind_speed = output[0].cpu().numpy()
                data.append({
                    'Altitude': altitude,
                    'Density': density,
                    'Pressure': pressure,
                    'Temperature': temperature,
                    'Speed of Sound': speed_of_sound,
                    'Wind Speed': wind_speed
                })
        
        return pd.DataFrame(data)
    
    def save_environment_data(self, filename='EnvironmentData.csv'):
        """Save environment data to CSV file"""
        folder_name = "Environment"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        csv_filename = os.path.join(folder_name, filename)
        
        df = self.generate_environment_data()
        df.to_csv(csv_filename, index=False)
        print(f"Environment data saved to {csv_filename}")
        
    def save_model(self, filename='atmospheric_model.pt'):
        """Save the trained PyTorch model"""
        folder_name = "Environment"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        model_filename = os.path.join(folder_name, filename)
        torch.save(self.model.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate environment data for rocket simulation")
    parser.add_argument("latitude", type=float, help="Latitude of the launch site (degrees)")
    parser.add_argument("longitude", type=float, help="Longitude of the launch site (degrees)")
    parser.add_argument("launch_altitude", type=float, help="Launch altitude of the site (meters)")
    parser.add_argument("--train", action="store_true", help="Train the neural network model")
    args = parser.parse_args()
    env_gen = EnvironmentGenerator(args.latitude, args.longitude, args.launch_altitude)
    
    if args.train:
        print("Training neural network model...")
        env_gen.train_model()
        env_gen.save_model()
    
    print("Generating environment data...")
    env_gen.save_environment_data()
    
    print("Environment generation complete!")

if __name__ == "__main__":
    main()