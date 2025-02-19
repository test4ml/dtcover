

from enum import Enum

class MutationResult(Enum):
    MUTATED = "MUTATED"
    DUMP = "DUMP"
    CONFLICT = "CONFLICT"
    NER = "NER"