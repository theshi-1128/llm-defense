import os
import time
import pandas as pd


class IntervalSaver:
    def __init__(self, output_path, interval=1 * 1 * 60, columns=['prompt', 'response', 'judgement']):
        """
        Initialize the IntervalSaver class.

        :param output_path: The path for the output file.
        :param interval: The interval time for saving in seconds.
        :param columns: The column names for the DataFrame.
        """
        self.output_path = output_path
        self.interval = interval
        self.start_time = time.time()
        self.results_df = pd.DataFrame(columns=columns)

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir) and output_dir:
            os.makedirs(output_dir)

    def add_and_save(self, row):
        """
        Add a row of data to the DataFrame and save it to a file after a specified time interval.

        :param row: The data to be added, should be in the form of a dictionary.
        """
        current_time = time.time()
        if current_time - self.start_time >= self.interval:
            self.save_results()
            self.start_time = time.time()

        new_row = pd.DataFrame([row])  # Simplified DataFrame creation
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)

    def save_results(self):
        """
        Save the DataFrame to a CSV file, and print the saving time and the number of rows processed.
        """
        self.results_df.to_csv(self.output_path, index=False)
        print(f"Temporary save at {time.ctime()} after processing {len(self.results_df)} rows.")

    def final_save(self):
        """
        Finally, save the DataFrame to a CSV file, to be called at the end of the program.
        """
        self.save_results()
        print(f"Responses saved to {self.output_path}.")
