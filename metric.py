from collections import Counter


def _lcs_len(a, b):
    dp = [[0] * (len(b) + 1)] * (len(a) + 1)
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


def rouge_L(output, reference, mode='f'):
    assert mode in list('fpr')  # F-1, precision, recall
    lcs = _lcs_len(output, reference)
    if lcs == 0:
        return 0.0
    else:
        precision = lcs / len(output)
        recall = lcs / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            return precision
        elif mode == 'r':
            return recall
        else:
            return f_score


def make_n_grams(seq, n):
    ngrams = (tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
    return ngrams


def n_gram_match(summ, ref, n):
    summ_ngrams = Counter(make_n_grams(summ, n))
    ref_ngrams = Counter(make_n_grams(ref, n))
    ngrams = min(summ_ngrams, ref_ngrams, key=len)
    count = sum(min(summ_ngrams[g], ref_ngrams[g]) for g in ngrams)
    return count


def rouge_n(output, reference, n=1, mode='f'):
    assert mode in list('fpr')
    match = n_gram_match(output, reference, n)
    if match == 0:
        return 0.0
    else:
        precision = match / len(output)
        recall = match / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            return precision
        elif mode == 'r':
            return recall
        else:
            return f_score
