import nltk
import re

emo_repl = {
    # good emotions
    "&lt;3": " good ",
    ":d": " good ",
    ":dd": " good ",
    ":p": " good ",
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",

    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    # bad emotions
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":s": " bad ",
    ":-s": " bad ",

    # good emotions
    "&lt;3": " heart ",
     ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",

    # bad emotions
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad "

}

emo_repl2 = {

}

emo_repl_order = [k for (k_len, k) in reversed(sorted([(len(k), k) for k in list(emo_repl.keys())]))]
emo_repl_order2 = [k for (k_len, k) in reversed(sorted([(len(k), k) for k in list(emo_repl2.keys())]))]


def replace_emojis(tweet):
    tweet2 = tweet
    for k in emo_repl_order:
        tweet2 = tweet2.replace(k, emo_repl[k])
    return tweet2


def replace_reg(tweet):
    tweet2 = tweet
    for k in emo_repl_order2:
        tweet2 = tweet2.replace(k, emo_repl2[k])
    return tweet2
