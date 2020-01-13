from .dataset import Dataset, MultiLabelEncoder, LabelEncoder
from typing import Tuple, Dict, List
from collections import defaultdict
import torch
import logging
import json


from pie import torch_utils


class ContextEncoder(LabelEncoder):
    def __init__(self, tokenizer: str, model_path: str, name: str = "context"):
        """

        :param self:
        :param tokenizer:
        :param model_path:
        :param name:
        :return:
        """
        import transformers
        self.type = tokenizer
        self.model_path = model_path
        self.tokenizer: transformers.PreTrainedTokenizer = getattr(transformers, tokenizer).from_pretrained(model_path)
        self.start_token = self.tokenizer.bos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.name = name

    def jsonify(self):
        return {
            "tokenizer": self.type,
            "model_path": self.model_path,
            "name": self.name
        }

    def get_pad(self):
        return self.tokenizer.pad_token_id

    def transform(self, seq: List[str]):
        encoded = self.tokenizer.encode([self.start_token] + seq, add_special_tokens=False)
        assert len(seq) + 1 == len(encoded), "Length of encoded and input should not vary, except for" \
                                                          "sentence beginning"
        return encoded


class TransformerMultiLabelEncoder(MultiLabelEncoder):
    def __init__(self, context_kwargs: Dict[str, str], word_max_size=None, char_max_size=None,
                 word_min_freq=1, char_min_freq=None, char_eos=True, char_bos=True):
        super(TransformerMultiLabelEncoder, self).__init__(
            word_max_size=word_max_size,
            char_max_size=char_max_size,
            word_min_freq=word_min_freq,
            char_min_freq=char_min_freq,
            char_eos=char_eos,
            char_bos=char_bos
        )
        self.context = ContextEncoder(**context_kwargs)

    def transform(self, sents):
        # Repeating the code, even though it'd make more sense to reuse, in the end we avoid repeating the loop
        word, char, context, tasks_dict = [], [], [], defaultdict(list)

        for inp in sents:
            tasks = None

            # task might not be passed
            if isinstance(inp, tuple):
                inp, tasks = inp

            # input data
            word.append(self.word.transform(inp))
            context.append(self.context.transform(inp))
            for w in inp:
                char.append(self.char.transform(w))

            # task data
            if tasks is None:
                # during inference there is no task data (pass None)
                continue

            for le in self.tasks.values():
                task_data = le.preprocess(tasks[le.target], inp)
                # add data
                if le.level == 'token':
                    tasks_dict[le.name].append(le.transform(task_data))
                elif le.level == 'char':
                    for w in task_data:
                        tasks_dict[le.name].append(le.transform(w))
                else:
                    raise ValueError("Wrong level {}: task {}".format(le.level, le.name))

        return (word, char, context), tasks_dict

    @classmethod
    def from_settings(cls, settings, tasks=None):
        le = cls(
            context_kwargs={
                "tokenizer": settings.transformer["tokenizer_class"],
                "model_path": settings.transformer["tokenizer_path"]
            },
            word_max_size=settings.word_max_size,
            word_min_freq=settings.word_min_freq,
            char_max_size=settings.char_max_size,
            char_min_freq=settings.char_min_freq,
            char_eos=settings.char_eos,
            char_bos=settings.char_bos
        )

        for task in settings.tasks:
            if tasks is not None and task['settings']['target'] not in tasks:
                logging.warning(
                    "Ignoring task [{}]: no available data".format(task['target']))
                continue
            le.add_task(task['name'], level=task['level'], **task['settings'])

        return le

    def jsonify(self):
        return {'word': self.word.jsonify(),
                'char': self.char.jsonify(),
                'context': self.context.jsonify(),
                'tasks': {le.name: le.jsonify() for le in self.tasks.values()}}

    def save(self, path):
        with open(path, 'w+') as f:
            json.dump(self.jsonify(), f)

    @staticmethod
    def _init(inst, obj):
        inst.word = LabelEncoder.from_json(obj['word'])
        inst.char = LabelEncoder.from_json(obj['char'])

        for task, le in obj['tasks'].items():
            inst.tasks[task] = LabelEncoder.from_json(le)

        return inst

    @classmethod
    def load_from_string(cls, string):
        obj = json.loads(string)
        inst = cls(context_kwargs=obj["context"])
        return cls._init(inst, obj)


class TransformerDataset(Dataset):
    def pack_batch(self, batch, device=None) -> Tuple[Tuple[torch.tensor, torch.tensor, torch.tensor], torch.tensor]:
        """
        Transform batch data to tensors
        """
        (word, char, context), tasks = self.label_encoder.transform(batch)
        word = torch_utils.pad_batch(word, self.label_encoder.word.get_pad(), device=device)
        char = torch_utils.pad_batch(char, self.label_encoder.char.get_pad(), device=device)
        context = torch_utils.pad_batch(context, self.label_encoder.context.get_pad(), device=device)

        output_tasks = {}
        for task, data in tasks.items():
            output_tasks[task] = torch_utils.pad_batch(
                data, self.label_encoder.tasks[task].get_pad(), device=device)

        return (word, char, context), output_tasks

