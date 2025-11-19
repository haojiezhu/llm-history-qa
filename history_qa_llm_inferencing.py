import os, random
from ollama import Client

os.system("./ollama/bin/ollama pull gpt-oss:20b")
client = Client()

qa_dataset = []

file_path = "./history_qa_csv.csv"
with open(file_path, 'r') as file:
	lines = file.readlines()
	for line in lines:
		data_fields = line.strip().split("|")
		if data_fields[0] != "Template_ID":
			qa_template = data_fields[1]
			qa_id = data_fields[2]
			num_placeholders = int(data_fields[3])
			num_choices = int(data_fields[4])
			ground_truth = data_fields[5]
			# Parse all MCQ choices
			all_choices = data_fields[6].strip().split(";")
			for i in range(len(all_choices)):
				all_choices[i] = all_choices[i].strip()
			# Randomizing choices and change ground_truth accordingly
			if num_choices == 4:
				# Example: "C" is ground truth -> ground_truth_index = 2
				dict_letter_index = {"A":0, "B":1, "C":2, "D":3}
				ground_truth_index = dict_letter_index[ground_truth]
				indices = [0, 1, 2, 3]
				random.shuffle(indices)
				shuffled_choices = [all_choices[indices[0]], all_choices[indices[1]], all_choices[indices[2]], all_choices[indices[3]]]
				# Find new ground truth position after shuffling, e.g. [1, 3, 0, 2]
				ground_truth_index_new = indices.index(ground_truth_index)
				dict_index_letter = {0:"A", 1:"B", 2:"C", 3:"D"}
				# Update ground truth
				ground_truth = dict_index_letter[ground_truth_index_new]
				# Update all choices
				all_choices = shuffled_choices
			# Insert placeholders into template
			for i in range(num_placeholders):
				str_to_replace = "[p" + str(i+1) + "]"
				qa_template = qa_template.replace(str_to_replace, data_fields[i+7])
			#print("{0}, {1}".format(qa_id, qa_template))
			#print("{0}, {1}".format(ground_truth, all_choices))
			# Construct final LLM QA question
			if num_choices == 4:
				prefix = qa_id + " | You are an expert on historical events. Please select the best choice for the following multiple choice question. Don\'t show any reasoning process. Just give me the letter of your best choice.\n"
				suffix = "\nA. " + all_choices[0] + "\nB. " + all_choices[1] + "\nC. " + all_choices[2] + "\nD. " + all_choices[3] + " | " + ground_truth
				llm_qa_question = prefix + qa_template + suffix
			elif num_choices == 2 and ground_truth in ["TRUE", "FALSE"]:
				if "Verify the claim" in qa_template:
					prefix = qa_id + " | You are an expert on historical events and tasked with verifying whether the hypothetical scenario in the following claim can possibly happen. Don\'t show any reasoning process. Just give me TRUE or FALSE as your answer. Here is the claim: "
					suffix = " | " + ground_truth
					llm_qa_question = prefix + qa_template + suffix
				else:
					prefix = qa_id + " | You are an expert on historical events. Please answer the following true or false question. Don\'t show any reasoning process. Just give me TRUE or FALSE as your answer. Here is the question: "
					suffix = " | " + ground_truth
					llm_qa_question = prefix + qa_template + suffix
			elif num_choices == 2 and ground_truth in ["A", "B"]:
				llm_qa_question = "N/A"
			#print(llm_qa_question)
			if llm_qa_question != "N/A":
				qa_dataset.append(llm_qa_question);

list_results = []

for qa_row in qa_dataset:
	qa_question = qa_row.strip().split("|")
	resp_fast = client.chat(model="gpt-oss:20b", messages=[{"role": "user", "content": qa_question[1]}], think="low")
	list_results.append({"question_id": qa_question[0], "question_text": qa_question[1], "ground_truth": qa_question[2], "model_inference": resp_fast["message"]["content"]})

print("question_id" + "	| " + "question_text")
for result in list_results:
	print("-------------------------------")
	print(result["question_id"] + "		| " + result["question_text"])

print()

print("question_id" + "	| " + "ground_truth" + "	| " + "model_inference")
for result in list_results:
	print(result["question_id"] + "		| " + result["ground_truth"] + "		| " + result["model_inference"])
