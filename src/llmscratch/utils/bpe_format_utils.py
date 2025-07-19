from typing import Dict, Tuple


def convert_to_sennrich_format(
    vocab: Dict[Tuple[str, ...], int],
) -> Dict[Tuple[str, ...], int]:
    """
    Converts vocab from paper-faithful format:
        ('v', 'i', 'l', 'l', 'a', '</w>')
    to Sennrich GitHub format:
        ('v', 'i', 'l', 'l', 'a</w>')
    """
    converted = {}
    for symbols, freq in vocab.items():
        if symbols[-1] == "</w>":
            last = symbols[-2] + "</w>"
            converted[tuple(symbols[:-2] + (last,))] = freq
        else:
            converted[symbols] = freq
    return converted


def convert_from_sennrich_format(
    vocab: Dict[Tuple[str, ...], int],
) -> Dict[Tuple[str, ...], int]:
    """
    Converts vocab from Sennrich GitHub format:
        ('v', 'i', 'l', 'l', 'a</w>')
    to paper-faithful format:
        ('v', 'i', 'l', 'l', 'a', '</w>')
    """
    converted = {}
    for symbols, freq in vocab.items():
        last = symbols[-1]
        if last.endswith("</w>"):
            base = last[:-4]
            converted[tuple(symbols[:-1] + (base, "</w>"))] = freq
        else:
            converted[symbols] = freq
    return converted
