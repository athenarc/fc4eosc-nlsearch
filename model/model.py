import torch
import pandas as pd
from typing import List
from transformers import T5ForConditionalGeneration, AutoConfig, AutoTokenizer


# class T5Dataset(torch.utils.data.Dataset):
#     def __init__(self, data, tokenizer):
#         '''
#         Creates Dataset instance, designed for use with the T5 model.

#             Parameters:
#                 data (DataFrame): A pandas DataFrame with three columns named
#                 'prefix', 'input_text', 'target_text'.
#                 tokenizer: The appropriate tokenizer for the model.
#         '''

#         self.data = data
#         self.tokenizer = tokenizer

#     def _create_encodings(self, idx):
#         # Get data at given idx
#         inputs = self.data['prefix'][idx] + ": " + self.data['input_text'][idx]
#         targets = self.data['target_text'][idx]

#         # Convert to list if we have more than one example
#         if not (isinstance(inputs, str) and isinstance(targets, str)):
#             inputs = inputs.to_list()
#             targets = targets.to_list()

#         input_encodings = self.tokenizer(
#             inputs, truncation=True, padding='max_length', return_tensors="pt")

#         with self.tokenizer.as_target_tokenizer():
#             target_encodings = self.tokenizer(
#                 targets, truncation=True, padding='max_length', return_tensors="pt")

#         return input_encodings, target_encodings

#     def __getitem__(self, idx):
#         inputs, targets = self._create_encodings(idx)
#         item = inputs
#         item['labels'] = targets['input_ids']
#         item = {key: torch.squeeze(val) for key, val in inputs.items()}

#         return item

#     def __len__(self):
#         return len(self.data)

class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        '''
        Creates Dataset instance, designed for use with the T5 model.

            Parameters:
                data (DataFrame): A pandas DataFrame with three columns named
                'prefix', 'input_text', 'target_text'.
                tokenizer: The appropriate tokenizer for the model.
        '''

        self.data = data
        self.tokenizer = tokenizer

        self._create_encodings()

    def _create_encodings(self):
        inputs = (self.data['prefix'] + ": " +
                  self.data['input_text']).to_list()
        self.input_encodings = self.tokenizer(
            inputs, truncation=True, padding=True, return_tensors="pt")

        targets = self.data['target_text'].to_list()
        with self.tokenizer.as_target_tokenizer():
            self.target_encodings = self.tokenizer(
                targets, truncation=True, padding=True, return_tensors="pt")

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.input_encodings.items()}
        item['labels'] = self.target_encodings['input_ids'][idx].clone().detach()
        return item

    def __len__(self):
        return len(self.data)


def load_model(model_name_or_path='t5-base'):
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, config=config)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    return model, tokenizer


def generate_outputs(model, tokenizer, input_text, prefix, k=4, do_sample=False, num_beams=4):
    input_ids = tokenizer(f"{prefix}{input_text}",
                          return_tensors="pt").input_ids
    generated_outputs = model.generate(input_ids, max_length=512, do_sample=do_sample, num_beams=num_beams,
                                       num_return_sequences=k, output_scores=True, return_dict_in_generate=True)
    outputs = tokenizer.batch_decode(
        generated_outputs.sequences, skip_special_tokens=True)

    return outputs


def batch_generate_outputs(model, tokenizer, input_texts, k=4, do_sample=False, num_beams=4):
    inputs = tokenizer(input_texts, padding=True,
                       truncation=True, return_tensors="pt")

    batch_outputs = model.generate(**inputs, max_length=512, do_sample=do_sample, num_beams=num_beams,
                                   num_return_sequences=k, output_scores=True, return_dict_in_generate=True)

    outputs = tokenizer.batch_decode(
        batch_outputs.sequences, skip_special_tokens=True)

    return outputs


def remove_db_ids(predictions: List[str]):
    """Removes DB ids from predictions, for models that follow this format

    Args:
        predictions (List[str]): A list of predictions from a model that begins its predictions with the DB id
    """

    ret = []
    for prediction in predictions:
        db_id, sql = prediction.split(sep=" | ", maxsplit=1)
        ret.append(sql)

    return ret


def calculate_probabilities(generated_outputs):
    """
    Calculates the probabilities with which each token was generated
    Modified from the following post:
    https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175
    NOTE: This function only works when using num_beams=1

    Args:
        generated_outputs : The dictionary output of the model.generate function
    """

    # Discard the initial <pad> token
    gen_sequences = generated_outputs.sequences[:, 1:]  # shape [k, max_len]

    # let's stack the logits generated at each step to a tensor and transform
    # logits to probs
    # shape [k, max_len, vocab_size]
    probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)

    # now we need to collect the probability of the generated token
    # we need to add a dummy dim in the end to make gather work
    gen_probs = torch.gather(
        probs, 2, gen_sequences[:, :, None]).squeeze(-1)  # shape [k, max_len]

    # now we can do all kinds of things with the probs

    # 1) the probs that exactly those sequences are generated again
    # those are normally going to be very small
    unique_prob_per_sequence = gen_probs.prod(-1)

    # 2) normalize the probs over the three sequences
    normed_gen_probs = gen_probs / gen_probs.sum(0)
    assert normed_gen_probs[:, 0].sum() == 1.0, "probs should be normalized"

    # 3) compare normalized probs to each other like in 1)
    unique_normed_prob_per_sequence = normed_gen_probs.prod(-1)

    return unique_prob_per_sequence
