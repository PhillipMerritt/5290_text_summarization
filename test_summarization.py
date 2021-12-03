import os

import datasets
from datasets import load_dataset
from transformers import (BartForConditionalGeneration, 
                            BertTokenizer,
                            BartTokenizer,
                            Trainer,
                            TrainingArguments,
                            trainer_seq2seq,
                            training_args_seq2seq,
                            EncoderDecoderModel)
from transformers.trainer_utils import EvaluationStrategy
#from transformers.utils.dummy_pt_objects import Seq2SeqTrainer

model_directory = 'data/models/bart_attempt_3/'#'./data/models/bert2bert/'
checkpoint = 'data/models/bart_attempt_3/checkpoint-60000'#'data/models/test_summarization_bart1/checkpoint-23000'#'facebook/bart-base'#'facebook/bert-base-uncased'
""" checkpoints = [name for name in os.listdir(model_directory) if name.startswith('checkpoint')]

if checkpoints:
    checkpoints = sorted(checkpoints)
    print('using checkpoint: {}'.format(checkpoints[-1]))
    checkpoint = model_directory + checkpoints[-1] """

if 'bart' in checkpoint:
    tokenizer = BartTokenizer.from_pretrained(checkpoint)
else:
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

encoder_max_length=512
decoder_max_length=128
batch_size = 8

# credit to https://github.com/patrickvonplaten/notebooks for this cnn processing function
def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

if __name__ == '__main__':

    rouge = datasets.load_metric("rouge")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    val_dataset = load_dataset("cnn_dailymail", '2.0.0', split='validation[:5%]')

    print('tokenizing val...')
    tokenized_val = val_dataset.map(process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["article", "highlights", "id"],
        num_proc=4)
    tokenized_val.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    if 'bert' in checkpoint:
        model = EncoderDecoderModel.from_pretrained(checkpoint)

        # set special tokens
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        # sensible parameters for beam search
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.max_length = 142
        model.config.min_length = 56
        model.config.no_repeat_ngram_size = 3
        model.config.early_stopping = True
        model.config.length_penalty = 2.0
        model.config.num_beams = 4
    else:
        model = BartForConditionalGeneration.from_pretrained(checkpoint)

    training_args = training_args_seq2seq.Seq2SeqTrainingArguments(model_directory, 
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=batch_size, 
        num_train_epochs=2, 
        evaluation_strategy='epoch',
        fp16=True,
        save_steps=1000,
        do_train=False,
        do_eval=True,
        predict_with_generate=True
        )

    trainer = trainer_seq2seq.Seq2SeqTrainer(
        model,
        training_args,
        #train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        #data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    eval_output = trainer.evaluate()
    print(eval_output)
