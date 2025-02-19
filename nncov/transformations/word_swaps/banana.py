from textattack.transformations import WordSwap


class BananaWordSwap(WordSwap):
    """Transforms an input by replacing any word with 'banana'."""

    # We don't need a constructor, since our class doesn't require any parameters.

    def _get_replacement_words(self, word):
        """Returns 'banana', no matter what 'word' was originally.

        Returns a list with one item, since `_get_replacement_words` is intended to
            return a list of candidate replacement words.
        """
        return ["banana"]
