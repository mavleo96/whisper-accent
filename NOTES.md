1. Accent Tokens
- [DONE] _retrieve_init_tokens -> detect_accent -> add before timestamps token
- Forward method -> should not use this
- change this to insert before timestamps token

2. Tokenizer
- prefix tokens should include mainstream english accent token -> prefix_tokens property & set_prefix_tokens method
- modify batch_encode_plus and encode_plus to insert accent tokens before timestamps token
