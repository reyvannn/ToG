import argparse
from utils import *
import json

def jsonl_to_json(jsonl_file, json_file):
    with open(jsonl_file, 'r') as infile:
        with open(json_file, 'w') as outfile:
            json_lines = infile.readlines()
            json_list = [json.loads(line) for line in json_lines]
            json.dump(json_list, outfile, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--output_file", type=str,
                        default="ToG_cwq.json", help="the output file name.")
    parser.add_argument("--constraints_refuse", type=bool,
                        default=True, help="LLM may have refuse erorr, enable this option to skip current sample.")
    args = parser.parse_args()

    if str(args.output_file).endswith(".jsonl"):
        jsonl_to_json(args.output_file, args.output_file[:-1])
        args.output_file = args.output_file[:-1]

    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(args.dataset, args.output_file)

    num_right = 0
    num_error = 0
    for data in output_datas:
        answers = align(args.dataset, question_string, data, ground_truth_datas)
        results = data['results']
        if check_string(results):
            response = clean_results(results)
            if response=="NULL":
                response = results
            else:
                if exact_match(response, answers):
                    num_right+=1
                else:
                    num_error+=1
        else:
            response = results
            if args.constraints_refuse and check_string(response):
                continue
            if exact_match(response, answers):
                num_right+=1
            else:
                num_error+=1

    print("Exact Match: {}".format(float(num_right/len(output_datas))))
    print("right: {}, error: {}".format(num_right, num_error))

    save_result2json(args.dataset, num_right, num_error, len(output_datas))
