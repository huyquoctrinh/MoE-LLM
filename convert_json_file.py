from smoe.utils.io import load_json, dump_jsonlines

data = load_json("resources/OpenHermes-2.5/openhermes2_5.json")
dump_jsonlines(data, "resources/OpenHermes-2.5/openhermes2_5.jsonl")