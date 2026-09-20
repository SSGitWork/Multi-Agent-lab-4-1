def count_words(text: str) -> dict[str, int]:
    """
    Counts the frequency of each word in the given text string.

    Args:
        text (str): The input text to count words from.

    Returns:
        dict[str, int]: A dictionary mapping words to their frequency counts.
    """
    import re
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq: dict[str, int] = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq