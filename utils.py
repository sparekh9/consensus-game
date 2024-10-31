import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def qadict_to_prompt(qa, subject=None, correct=True, answer=None):
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
    prompt += " {}".format(answer)
  
  return prompt