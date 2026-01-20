import sys
import os

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import train_model
from src.predict import run_weekly_prediction

def main():
    while True:
        print("\n=== ML Swing Trading System ===")
        print("1. Download Data & Train Model")
        print("2. Run Weekly Prediction")
        print("3. Exit")
        
        choice = input("Select an option (1-3): ")
        
        if choice == '1':
            train_model()
        elif choice == '2':
            run_weekly_prediction()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
