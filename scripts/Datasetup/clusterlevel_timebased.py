import pandas as pd
import yaml
import os

def load_config():
    """Load configuration from YAML file"""
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, '..', 'config', 'cluster_timebased.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def ensure_folder_exists(folder_path):
    """Create folder if it doesn't exist"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Created folder: {folder_path}")

def main():
    # Load configuration
    config = load_config()
    
    # Build file paths
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_root, config['input']['file'])
    output_folder = os.path.join(project_root, config['output']['folder'])
    output_file = os.path.join(output_folder, config['output']['filename'])
    
    # Ensure output folder exists
    ensure_folder_exists(output_folder)
    
    # Read data
    print(f"📖 Reading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Perform aggregation based on config
    aggregation_config = config['aggregation']['metrics']
    time_aggregated = df.groupby('ts').agg(aggregation_config).reset_index()
    
    # Keep only specified columns
    if 'columns' in config and 'keep' in config['columns']:
        time_aggregated = time_aggregated[config['columns']['keep']]
    
    # Save result
    time_aggregated.to_csv(output_file, index=False)
    
    print(f"✅ Aggregation completed!")
    print(f"📁 Output: {output_file}")
    print(f"📊 Result shape: {time_aggregated.shape}")
    print(f"📈 Columns: {', '.join(time_aggregated.columns)}")
    print("\nFirst 3 rows:")
    print(time_aggregated.head(3))

if __name__ == "__main__":
    main()