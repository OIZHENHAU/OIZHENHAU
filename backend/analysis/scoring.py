import numpy as np


def compute_df_stats(df) -> dict:
    """Pre-compute dataset percentiles used for reason generation."""
    stats = {}
    for col in ['engagement_rate', 'toxicity_score', 'like_rate',
                'share_rate', 'comment_rate', 'username_randomness']:
        q = np.nanpercentile(df[col].values, [10, 25, 75, 90])
        stats[col] = {'q10': q[0], 'q25': q[1], 'q75': q[2], 'q90': q[3],
                      'mean': float(df[col].mean())}
    bz = df['buzz_change_rate'].abs()
    stats['buzz_abs'] = {'q75': float(np.nanpercentile(bz.values, 75)),
                         'q90': float(np.nanpercentile(bz.values, 90))}
    return stats


def post_reasons(row: dict, stats: dict) -> list:
    """
    Return up to 2 reasons why a post/account was flagged.
    row must contain engagement_rate, toxicity_score, like_rate, buzz_change_rate, username_randomness.
    """
    signals = []

    er = row.get('engagement_rate', None)
    if er is not None:
        if er < stats['engagement_rate']['q10']:
            signals.append((
                abs(er - stats['engagement_rate']['mean']),
                {'title': 'Low engagement rate',
                 'description': f'{er:.4f} — far below average',
                 'icon': 'chart-line'}
            ))
        elif er > stats['engagement_rate']['q90']:
            signals.append((
                er - stats['engagement_rate']['q90'],
                {'title': 'Suspiciously high engagement',
                 'description': f'{er:.4f} — abnormally inflated',
                 'icon': 'chart-line'}
            ))

    tox = row.get('toxicity_score', None)
    if tox is not None and tox > stats['toxicity_score']['q75']:
        signals.append((
            tox - stats['toxicity_score']['q75'],
            {'title': 'High toxicity score',
             'description': f'{tox:.3f} — bot-like comment pattern',
             'icon': 'alert'}
        ))

    lr = row.get('like_rate', None)
    if lr is not None and lr > stats['like_rate']['q90']:
        signals.append((
            lr - stats['like_rate']['q90'],
            {'title': 'Abnormal like ratio',
             'description': f'{lr:.4f} — inflated like pattern',
             'icon': 'bar-chart'}
        ))

    bz = row.get('buzz_change_rate', None)
    if bz is not None and abs(bz) > stats['buzz_abs']['q90']:
        signals.append((
            abs(bz) - stats['buzz_abs']['q90'],
            {'title': 'Abnormal buzz change',
             'description': f'{bz:.1f} — unusual spike pattern',
             'icon': 'trending'}
        ))

    ur = row.get('username_randomness', None)
    if ur is not None and ur > 0.82:
        signals.append((
            ur - 0.82,
            {'title': 'High username randomness',
             'description': f'{ur:.2f} — auto-generated pattern',
             'icon': 'user'}
        ))

    if not signals:
        signals.append((1.0, {
            'title': 'Multivariate anomaly',
            'description': 'Subtle combination of signals detected',
            'icon': 'radar'}
        ))

    signals.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in signals[:2]]


def account_reasons(inputs: dict) -> list:
    """Generate reasons for the single-account-lookup result."""
    signals = []

    followers = inputs.get('followers', 0)
    following = inputs.get('following', 1)
    ff = followers / max(following, 1)
    if ff < 0.1:
        signals.append((0.1 - ff, {
            'title': 'Abnormal FF ratio',
            'description': f'{ff:.3f} — following far exceeds followers',
            'icon': 'bar-chart'}
        ))

    spam = inputs.get('spam_comments_rate', 0)
    if spam > 0.6:
        signals.append((spam - 0.6, {
            'title': 'High spam comment rate',
            'description': f'{spam:.2f} — bot-like posting pattern',
            'icon': 'alert'}
        ))

    generic = inputs.get('generic_comment_rate', 0)
    if generic > 0.6:
        signals.append((generic - 0.6, {
            'title': 'Generic comment pattern',
            'description': f'{generic:.2f} — homogeneous engagement',
            'icon': 'chart-line'}
        ))

    urand = inputs.get('username_randomness', 0)
    if urand > 0.7:
        signals.append((urand - 0.7, {
            'title': 'High username randomness',
            'description': f'{urand:.2f} — auto-generated pattern',
            'icon': 'user'}
        ))

    bio = inputs.get('bio_length', 50)
    if bio < 15:
        signals.append((15 - bio, {
            'title': 'Minimal bio',
            'description': f'{bio} chars — incomplete profile',
            'icon': 'user'}
        ))

    has_pic = inputs.get('has_profile_picture', 1)
    if has_pic == 0:
        signals.append((1.0, {
            'title': 'No profile picture',
            'description': 'Missing profile image — common bot signal',
            'icon': 'user'}
        ))

    age = inputs.get('account_age_days', 100)
    if age < 30:
        signals.append((30 - age, {
            'title': 'Very new account',
            'description': f'{age} days old — recently created',
            'icon': 'trending'}
        ))

    if not signals:
        signals.append((1.0, {
            'title': 'Multivariate anomaly',
            'description': 'Subtle combination of signals detected',
            'icon': 'radar'}
        ))

    signals.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in signals[:2]]
