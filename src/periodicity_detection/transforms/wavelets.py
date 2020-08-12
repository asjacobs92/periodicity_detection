""" Wavelets collection module """


def haar():
    """
        Generates low pass and high pass filters for the Haar wavelet.
        Returns:
            * lp_filter
                Low pass filter vector (a.k.a. wavelet function or mother wavelet)
            * hp_filter
                High pass filter vector (a.k.a. scaling function or father wavelet)
    """
    # return daubechies(1)
    return ([0.5,  0.5], [-0.5, 0.5])
#


def daubechies(scale=2):
    """
        Generates low pass and high pass filters for the Daubechies wavelet at a given scale.
        Args:
            - scale
                Desired scale for the filter functions.
                Currently supports: 1, 2, 3 and 4.
                Default: 2
        Returns:
            * lp_filter
                Low pass filter vector (a.k.a. wavelet function or mother wavelet)
            * hp_filter
                High pass filter vector (a.k.a. scaling function or father wavelet)
    """

    db_scales = {
        1: {
            'lp': [0.7071067812,  0.7071067812],
            'hp': [-0.7071067812, 0.7071067812]
        },
        2: {
            'lp': [-0.1294095226, 0.2241438680, 0.8365163037, 0.4829629131],
            'hp': [-0.4829629131, 0.8365163037, -0.2241438680, -0.1294095226]
        },
        3: {
            'lp': [0.0352262919, -0.0854412739, -0.1350110200, 0.4598775021, 0.8068915093, 0.3326705530],
            'hp': [-0.3326705530, 0.8068915093, -0.4598775021, -0.1350110200, 0.0854412739, 0.0352262919]
        },
        4: {
            'lp': [-0.0105974018, 0.0328830117, 0.0308413818, -0.1870348117, -0.0279837694, 0.6308807679, 0.7148465706, 0.2303778133],
            'hp': [-0.2303778133, 0.7148465706, -0.6308807679, -0.0279837694, 0.1870348117, 0.0308413818, -0.0328830117, -0.0105974018]
        }
    }

    filters = db_scales.get(scale, None)
    return (filters['lp'], filters['hp']) if filters else (None, None)
