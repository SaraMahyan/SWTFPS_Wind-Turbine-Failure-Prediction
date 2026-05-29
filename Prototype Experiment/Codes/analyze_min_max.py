import pandas as pd

def analyze_min_max(file_path):
    print(f"Analyzing {file_path}...")
    try:
        # Read the dataset
        df = pd.read_csv(file_path)
        
        # Calculate min and max for each numeric column
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            print("No numeric columns found in the dataset.")
            return

        # Create a summary dataframe for nice display
        summary = pd.DataFrame({
            'Column Name': numeric_df.columns,
            'Min': numeric_df.min().values,
            'Max': numeric_df.max().values
        })
        
        # Reset index to clean up output
        summary = summary.set_index('Column Name')

        print("\n--- Minimum and Maximum Values ---")
        print(summary.to_string())
        
        # Optionally, save to a CSV
        output_file = "min_max_summary.csv"
        summary.to_csv(output_file)
        print(f"\nSummary saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    file_path = "sampled_wind_data.csv"
    analyze_min_max(file_path)
