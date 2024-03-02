# In-context learning

To use the API of openai, you should copy your api_key into the `base_addition_icl.py` and `addition_icl.py` first.

`addition_icl.py` and `base_addition_icl.py` provide codes where we filter out some in-context examples and return the contribution of these examples in `.plt` files. To run this code, you should first call the function `generate_test_sample` to generate enough candidates test samples.

To compare the in-context learning performance of `direct`, `scratchpad` and `rule-following`, you can run the code in `icl_learning.py` and set the task as `direct`, `scratchpad` and `rf`.