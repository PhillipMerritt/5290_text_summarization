import datasets
from datasets import load_dataset
import os
from transformers import (BartForConditionalGeneration, 
                            BartTokenizer, 
                            Trainer,
                            TrainingArguments,
                            trainer_seq2seq,
                            training_args_seq2seq)
from transformers.trainer_utils import EvaluationStrategy
from transformers import EncoderDecoderModel

model_directory = 'data/models/bart_attempt_3/'
checkpoint = 'facebook/bart-base'

# check for checkpoints
checkpoints = [name for name in os.listdir(model_directory) if name.startswith('checkpoint')]

if checkpoints:
    model_checkpoint = model_directory + sorted(checkpoints)[-1]

# load tokenizer and set some special tokens
tokenizer = BartTokenizer.from_pretrained(checkpoint)
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

# summarization hyper-params
encoder_max_length=512
decoder_max_length=128
batch_size = 4
tokenizer_processes = 4

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
    # function to appy rouge metric
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 8),
            "rouge2_recall": round(rouge_output.recall, 8),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 8),
        }
    train_dataset, val_dataset = load_dataset("cnn_dailymail", '2.0.0', split='train[:50%]'), load_dataset("cnn_dailymail", '2.0.0', split='validation[:25%]')

    # tokenize training and validation data
    print('tokenizing train...')
    tokenized_train = train_dataset.map(process_data_to_model_inputs, 
        batched=True, 
        batch_size=batch_size,
        remove_columns=["article", "highlights", "id"],
        num_proc=tokenizer_processes)
    tokenized_train.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

    print('tokenizing val...')
    tokenized_val = val_dataset.map(process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["article", "highlights", "id"],
        num_proc=tokenizer_processes)
    tokenized_val.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    # load model from checkpoint of huggingface publicly available models
    model = BartForConditionalGeneration.from_pretrained(checkpoint)
    
    # training arguments
    training_args = training_args_seq2seq.Seq2SeqTrainingArguments(model_directory, 
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=batch_size, 
        num_train_epochs=1, 
        #evaluation_strategy='steps',
        fp16=True,
        save_steps=500,
        do_train=True,
        do_eval=False,
        #eval_steps=50,
        predict_with_generate=True
        )

    # create trainer
    trainer = trainer_seq2seq.Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_train,
        #eval_dataset=tokenized_val,
        #data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # train model and save results
    train_output = trainer.train(resume_from_checkpoint = checkpoint)
    trainer.save()
