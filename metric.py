def _lcs_len(a, b):
    dp = [[0] * (len(b) + 1)] * (len(a) + 1)
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i]==b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[-1][-1]


def rouge_L(output, reference, mode='f'):
    assert mode in list('fpr')
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