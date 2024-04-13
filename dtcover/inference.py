
import textattack

#@profile
def lm_forward(text: str, model_wrapper: textattack.models.wrappers.ModelWrapper, output_attentions=True):
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    model_device = next(model.parameters()).device

    input_sentence = [text]
    # tokens = tokenizer.tokenize(input_sentence)
    encoded_sentence = tokenizer(
        input_sentence,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True
    ).to(model_device)

    lm_output = model.forward(
        input_ids=encoded_sentence["input_ids"],
        attention_mask=encoded_sentence["attention_mask"],
        use_cache=False,
        past_key_values=None,
        output_attentions=output_attentions,
    )
    return lm_output
