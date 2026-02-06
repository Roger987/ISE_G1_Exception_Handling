import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    # Load table
    try:
        df = pd.read_csv('analysis_csv/RQ2_table.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Expect columns: Technology, Projects (Total), Findings (Raw), Error (%)
    x = df['Technology']
    y = df['Files (Total)']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple

    plt.figure(figsize=(6, 4))
    bars = plt.bar(x, y, color=colors)
    plt.xlabel('Technology')
    plt.ylabel('Number of Files')
    plt.title('Total Files per Technology')
    plt.tight_layout()

    # Save and/or show
    plt.savefig('analysis_csv/RQ2_projects_bar.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
