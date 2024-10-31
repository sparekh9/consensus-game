import torch
import numpy as np
from datasets import load_dataset
from reco import ReCoTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "huggyllama/llama-7b"
# model_path = "meta-llama/Llama-3.1-8B"
generator_actions = ["A", "B", "C", "D"]
generator_states = ["correct", "incorrect"]

discriminator_lambda_values = [1e-3]
generator_lambda_values = [1e-3]

lambda_S, lambda_L = np.meshgrid(discriminator_lambda_values, generator_lambda_values)
lambda_S = lambda_S.reshape(-1)
lambda_L = lambda_L.reshape(-1)

state_prior = np.ones(len(generator_states)) / len(generator_states)

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
    prompt += "\n{}. {}".format(generator_actions[j], choice)
  
  if correct:
    prompt += "\nAnswer:"
  else:
    prompt += "\nIncorrect Answer:"
      
  if answer is not None:
    prompt += " {}\n".format(answer)
  
  return prompt

def init_gen_policy(qa, tokenizer, model):
  """
  Initialize the generator policy from a qadict.

  Parameters
  ----------
  qa : dict
    A qadict containing the question, choices, and answer.
  tokenizer : transformers.PreTrainedTokenizer
    The tokenizer to use for encoding the prompt.
  model : transformers.PreTrainedModel
    The model to use for generating text.

  Returns
  -------
  policy : torch.tensor
    The generator policy, a tensor of shape (2, 4) representing the probability of each choice given the correct/incorrect prompt.
  """
  policy = []
  # Conditional generation on correct
  prompt = qadict_to_prompt(qa, correct=True)

  input_ids = tokenizer(prompt, return_tensors='pt').to(device)
  outputs = model.generate(**input_ids, 
                          max_new_tokens=1, 
                          output_logits=True,
                          return_dict_in_generate=True,
                          pad_token_id=tokenizer.eos_token_id)
  gen_probs_correct = torch.nn.functional.softmax(torch.tensor([outputs.logits[0][0][i] for i in tokenised_choices]), dim=0)
  policy.append(gen_probs_correct)
  
  # Conditional generation on incorrect
  prompt = qadict_to_prompt(qa, correct=False)

  input_ids = tokenizer(prompt, return_tensors='pt').to(device)
  outputs = model.generate(**input_ids, 
                          max_new_tokens=1, 
                          output_logits=True,
                          return_dict_in_generate=True,
                          pad_token_id=tokenizer.eos_token_id)
  gen_probs_incorrect = torch.nn.functional.softmax(torch.tensor([outputs.logits[0][0][i] for i in tokenised_choices]), dim=0)
  policy.append(gen_probs_incorrect)
  probs = torch.stack(policy, dim=0)
  reg_probs = torch.nn.functional.normalize(probs, p=1, dim=0)
  return reg_probs

def init_disc_policy(qa, tokenizer, model):
  """
  Initialize the discriminator policy from a qadict.

  Parameters
  ----------
  qa : dict
    A qadict containing the question, choices, and answer.
  tokenizer : transformers.PreTrainedTokenizer
    The tokenizer to use for encoding the prompt.
  model : transformers.PreTrainedModel
    The model to use for generating text.

  Returns
  -------
  policy : torch.tensor
    The discriminator policy, a tensor of shape (4, 2) representing the probability of each correctness given each choice.
  """
 
  policy = []
  for choice in generator_actions:
    prompt = qadict_to_prompt(qa, correct=True, answer=choice)
    input_ids = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(**input_ids, 
                            max_new_tokens=1, 
                            output_logits=True,
                            return_dict_in_generate=True,
                            pad_token_id=tokenizer.eos_token_id)
    disc_probs = torch.nn.functional.softmax(torch.tensor([outputs.logits[0][0][i] for i in correctness_choices]), dim=0)
    policy.append(disc_probs)
  probs = torch.stack(policy, dim=0)
  reg_probs = torch.nn.functional.normalize(probs, p=1, dim=0)
  return reg_probs

def equilibrium_ranking(qa, tokenizer, model, use_avg=False, random_sample=False):
  """
  Compute the equilibrium ranking for a given question-answer pair using a generator and discriminator model.

  Parameters
  ----------
  qa : dict
    A question-answer dictionary containing the question, choices, and answer.
  tokenizer : transformers.PreTrainedTokenizer
    The tokenizer used for encoding the prompt.
  model : transformers.PreTrainedModel
    The model used for generating text.
  use_avg : bool, optional
    If True, use average strategies from training; otherwise, use the latest strategies. Default is False.
  random_sample : bool, optional
    If True, sample answers based on computed probabilities; otherwise, use the most probable answers. Default is False.

  Returns
  -------
  gen_answer : int
    The index of the generated answer.
  disc_answer : int
    The index of the discriminator's answer.
  """
  generator_priors = init_gen_policy(qa, tokenizer, model).detach().cpu().numpy()
  discriminator_priors = init_disc_policy(qa, tokenizer, model).detach().cpu().numpy()

  reco_trainer = ReCoTrainer(
    generator_states,
    generator_actions,
    state_prior,
    generator_priors,
    discriminator_priors,
    generator_lambda_param=lambda_S,
    discriminator_lambda_param=lambda_L,
    alternating=False,
    seed = 0
  )

  generator_avg_strategy, discriminator_avg_strategy = reco_trainer.train(5000)
  
  if use_avg:
    generator_strategy = generator_avg_strategy
    discriminator_strategy = discriminator_avg_strategy
  else:
    generator_strategy = reco_trainer.get_generator_reco_strategies()
    discriminator_strategy = reco_trainer.get_discriminator_reco_strategies()
      
  generator_probs = generator_strategy[0, 0]
  discriminator_policy = discriminator_strategy / np.sum(discriminator_strategy, axis=1, keepdims=True)
  discriminator_probs = discriminator_policy[0, :, 0]

  if random_sample:
    gen_answer = np.random.choice(len(generator_probs), p=generator_probs)
    disc_answer = np.random.choice(len(discriminator_probs), p=discriminator_probs)
  else:
    gen_answer = np.argmax(generator_probs)
    disc_answer = np.argmax(discriminator_probs)
      
  return gen_answer, disc_answer

if __name__ == "__main__":

  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto"
  )
  model.eval()
  
  tokenised_choices = tokenizer.encode(" ".join(generator_actions))[1:]
  correctness_choices = tokenizer.encode("correct incorrect")[1:]
  
  overall_correct = [0, 0]
  overall_len = 0
  ind_acc = []
  for subject in subjects:
      correct = [0, 0]
      print(subject)
      ds = load_dataset("cais/mmlu", subject, split="test")
      for qa in ds:
          gen_answer, disc_answer = equilibrium_ranking(qa, tokenizer, model)
          if (gen_answer == qa['answer']):
              correct[0] += 1
          if (disc_answer == qa['answer']):
              correct[1] += 1
      overall_len += len(ds)
      overall_correct[0] += correct[0]
      overall_correct[1] += correct[1]
      ind_acc.append([correct[i] / len(ds) for i in range(2)])
      print(ind_acc[-1])
  print("Gen Accuracy: ", (overall_correct[0] / overall_len), ", Disc Accuracy: ", (overall_correct[1] / overall_len))
  
  
  
  