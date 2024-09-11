import argparse
import warnings
from pipeline.pipeline_initialization import pipeline_initialization
from pipeline.pipeline_execution import pipeline_execution

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Process input parameters for generating text and judging harmfulness.')
    parser.add_argument('--target_model_name', type=str, default='glm4', help='Name of the target model')
    parser.add_argument('--assist_model_name', type=str, default='deberta', help='Name of the assist model')
    parser.add_argument('--moderation_model_name', type=str, default='llama-guard', help='Name of the moderation model')
    parser.add_argument('--zero_shot_model_name', type=str, default='zero-shot', help='Name of the zero-shot model')
    parser.add_argument('--target_model_cuda_id', type=str, default="cuda:0", help='CUDA device for the target model')
    parser.add_argument('--assist_model_cuda_id', type=str, default="cuda:1", help='CUDA device for the assist model')
    parser.add_argument('--moderation_model_cuda_id', type=str, default="cuda:4", help='CUDA device for the moderation model')
    parser.add_argument('--zero_shot_model_cuda_id', type=str, default="cuda:1", help='CUDA device for the zero-shot model')
    parser.add_argument('--save_interval', type=int, default=1 * 1 * 30, help='Interval of saving CSV file in seconds')
    parser.add_argument('--dataset_dir', type=str, default='/home/test.csv', help='Directory of the dataset')
    parser.add_argument('--output_dir', type=str, default='/home/llm_defense/result/output.csv', help='Directory of the result')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    # Initialize the pipeline with the parsed arguments
    initialize_data = pipeline_initialization(args)
    # Execute the pipeline
    pipeline_execution(**initialize_data)

