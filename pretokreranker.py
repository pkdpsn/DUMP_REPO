async def predict_batch(self, batch_inputs: List[Dict]):
    pairs_input_ids = []
    pairs_attention_mask = []

    for item in batch_inputs:
        query = item["query"]
        chunks = item["chunks"]  # Now: list of (pre_tokenized_ids, pre_tokenized_mask)

        # Tokenize query once per request
        query_inputs = self.model.tokenizer(
            query,
            padding=False,           # We'll pad later
            truncation=True,
            max_length=64,           # Queries are short
            return_tensors='pt'
        )

        for chunk_ids, chunk_mask in chunks:  # Pre-tokenized
            # Concat: [CLS] query [SEP] chunk [SEP] + padding
            full_ids = torch.cat([
                query_inputs['input_ids'],
                torch.tensor([[tokenizer.sep_token_id]]),  # [SEP]
                chunk_ids
            ], dim=1)

            full_mask = torch.cat([
                query_inputs['attention_mask'],
                torch.tensor([[1]]),
                chunk_mask
            ], dim=1)

            # Pad to max_length if needed (or pre-pad chunks to full)
            if full_ids.shape[1] < self.max_length:
                pad_len = self.max_length - full_ids.shape[1]
                full_ids = torch.cat([full_ids, torch.zeros((1, pad_len), dtype=torch.long)], dim=1)
                full_mask = torch.cat([full_mask, torch.zeros((1, pad_len), dtype=torch.long)], dim=1)

            pairs_input_ids.append(full_ids.squeeze(0))
            pairs_attention_mask.append(full_mask.squeeze(0))

    # Stack into batch
    batch_ids = torch.stack(pairs_input_ids).to(self.model.device)
    batch_mask = torch.stack(pairs_attention_mask).to(self.model.device)

    with torch.no_grad():
        logits = self.model.model(input_ids=batch_ids, attention_mask=batch_mask).logits
        scores = logits.squeeze(-1).cpu().numpy()

    # Rest of sorting logic same as before