import torch
import numpy as np
from datasets import load_dataset
from reco import ReCoTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# model_path = "huggyllama/llama-7b"
model_path = "meta-llama/Llama-3.1-8B"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
choices = ["A", "B", "C", "D"]
subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

def qadict_to_prompt(qa, subject=None, correct=True, answer=None):
  """
  Convert a qadict to a prompt string suitable for use in generating text.

  Parameters
  ----------
  qa : dict
    A qadict containing the question, choices, and answer.
  subject : str, optional
    The subject of the question. If not provided, it is taken from the qadict.
  correct : bool, optional
    Whether to generate the correct or incorrect answer. Default is True.
  answer : str, optional
    The answer to include in the prompt. If not provided, it is taken from the qadict.

  Returns
  -------
  prompt : str
    The prompt string.
  """
  if subject is None:
    subject = qa['subject']
  subject = subject.split("_")
  subject = " ".join(subject)
  if correct:
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(subject)
  else:
    prompt = "The following are multiple choice questions (with incorrect answers) about {}.\n\n".format(subject)
  prompt += "{}".format(qa['question'])
  
  for j, choice in enumerate(qa['choices']):
    prompt += "\n{}. {}".format(choices[j], choice)
  
  if correct:
    prompt += "\nAnswer:"
  else:
    prompt += "\nIncorrect Answer:"
      
  if answer is not None:
    prompt += " {}\n".format(answer)
  
  return prompt

def queryLM(qa, tokenizer, model, sample=False):
  """
  Query the language model for the probability of each answer given a question.

  Parameters
  ----------
  qa : dict
    A qadict containing the question, choices, and answer.
  tokenizer : transformers.PreTrainedTokenizer
    The tokenizer to use for encoding the prompt.
  model : transformers.PreTrainedModel
    The model to use for generating text.
  sample : bool, optional
    If True, sample from the computed probabilities; otherwise, return the most probable answer. Default is False.

  Returns
  -------
  answer_idx : int
    The index of the answer.
  """
  prompt = qadict_to_prompt(qa)
  
  input_ids = tokenizer(prompt, return_tensors='pt').to(device)
  outputs = model.generate(**input_ids, 
                          max_new_tokens=1, 
                          output_logits=True,
                          return_dict_in_generate=True,
                          pad_token_id=tokenizer.eos_token_id)
  
  probs = torch.nn.functional.softmax(torch.tensor([outputs.logits[0][0][i] for i in tokenised_choices]), dim=0)
  if sample:
    return torch.multinomial(probs, 1)
  else:
    return torch.argmax(probs)
  
if __name__ == "__main__":

  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto"
  )
  model.eval()
  
  tokenised_choices = tokenizer.encode(" ".join(choices))[1:]
  
  overall_correct = 0
  overall_len = 0
  ind_acc = []
  for subject in subjects:
      correct = 0
      print(subject)
      ds = load_dataset("cais/mmlu", subject, split="test")
      for qa in ds:
          answer = queryLM(qa, tokenizer, model)
          if (answer == qa['answer']):
              correct += 1
      overall_len += len(ds)
      overall_correct += correct
      ind_acc.append(correct / len(ds))
      print(ind_acc[-1])
  print("Accuracy: ", (overall_correct / overall_len))
  