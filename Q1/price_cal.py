import numpy as np


month = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
         "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


class Date:
    def __init__(self, day, month, year):
        self.day = day
        self.month = month
        self.year = year


def countLeapYears(d):
    years = d.year
    if(d.month <= 2):
        years -= 1
    return years//4 - years//100 + years//400


def get_diff(dt1, dt2):
    n1 = dt1.year * 365 + dt1.day
    for i in range(dt1.month - 1):
        n1 += month_days[i]
    n1 += countLeapYears(dt1)

    n2 = dt2.year * 365 + dt2.day
    for i in range(dt2.month - 1):
        n2 += month_days[i]
    n2 += countLeapYears(dt2)

    return n2 - n1


def cal_probs(r, sigma, num_steps):
    numu = np.exp((r*num_steps)/2) - (np.exp((-sigma)*(np.sqrt(num_steps/2))))
    den = np.exp(sigma*np.sqrt(num_steps/2)) - \
        np.exp((-sigma)*(np.sqrt(num_steps/2)))
    pu = numu/den
    pu = pu ** 2

    numd = np.exp(sigma*np.sqrt(num_steps/2)) - np.exp((r*num_steps)/2)
    pd = numd / den
    pd = pd ** 2

    return pu, pd


def find_diff(t_d, t_e):
    d1 = int(t_d[:2])
    d2 = int(t_e[:2])
    y1 = int(t_d[5:])
    y2 = int(t_e[5:])
    m1 = month[t_d[2:5]]
    m2 = month[t_e[2:5]]
    dt1 = Date(d1, m1, y1)
    dt2 = Date(d2, m2, y2)

    return get_diff(dt1, dt2)


def price_xyz(A_t_d, B, t_e, r, sigma, t_d, num_steps):
    """
    Build the price lattice and returns X_td.
    """
    total_days = find_diff(t_d, t_e)
    print(total_days)
    delta_t = total_days / num_steps
    pu, pd = cal_probs(r, sigma, delta_t)
    ju = np.exp(sigma*np.sqrt(delta_t))
    jd = 1/ju

    A_t = []
    A_t.append([A_t_d])

    for i in range(1, num_steps+1):
        prices = []
        a0 = A_t[i-1][0]
        prices.append(a0*ju)
        prices.append(a0)
        prices.append(a0*jd)
        for j in range(1, len(A_t[i-1])):
            prices.append(A_t[i-1][j] * jd)
        A_t.append(prices)

    mul = np.exp(-r*delta_t)
    X = []
    Xet = []
    for x in A_t[num_steps]:
        Xet.append(max(x - B, 0))
    X.append(Xet)
    for i in range(1, num_steps+1):
        Xt = []
        for j in range(1, len(X[i-1])-1):
            xtj = X[i-1][j-1] * pu
            xtj += X[i-1][j] * (1-pu-pd)
            xtj += X[i-1][j+1] * pd
            xtj *= mul
            Xt.append(xtj)
        X.append(Xt)

    return X[num_steps][0]


def volrisk_xyz(A_t_d, B, t_e, r, sigma, t_d, num_steps):
    """
    return dXtd / dsigma
    """
    total_days = find_diff(t_d, t_e)
    delta_t = total_days / num_steps
    pu, pd = cal_probs(r, sigma, delta_t)
    ju = np.exp(sigma*np.sqrt(delta_t))
    jd = 1/ju

    dju = np.sqrt(delta_t) * ju
    djd = -np.sqrt(delta_t) * jd

    dpu_num = np.sqrt(2*delta_t) * (np.exp((np.sqrt(delta_t)*sigma)/np.sqrt(2)) - 1) \
            * (np.exp((3*np.sqrt(delta_t)*sigma)/np.sqrt(2) + (r*delta_t)/2) + 
                np.exp((np.sqrt(delta_t)*sigma)/np.sqrt(2) + (r*delta_t)/2)
                - 2*np.exp(sigma*np.sqrt(delta_t*2)))
    dpu_den = (np.exp(sigma*np.sqrt(delta_t*2)) - 1)**3
    dpu = -dpu_num / dpu_den

    dpd_num = np.sqrt(2*delta_t)*(np.exp((sigma*np.sqrt(delta_t)))/np.sqrt(2) - np.exp((r*delta_t)/2)) * \
                (np.exp((r*delta_t)/2)*(-np.exp(sigma*np.sqrt(2*delta_t)) - 1)+2*np.exp((sigma*np.sqrt(delta_t))/np.sqrt(2))) * \
                    np.exp(sigma * np.sqrt(2*delta_t))
    dpd_den = (np.exp(sigma*np.sqrt(delta_t*2)) - 1)**3
    dpd = -dpd_num / dpd_den

    A_t = []
    A_t.append([A_t_d])

    dA_t = []
    dA_t.append([0])

    for i in range(1, num_steps+1):
        prices = []
        dprices = []
        a0 = A_t[i-1][0]
        d0 = dA_t[i-1][0]

        prices.append(a0*ju)
        dprices.append(d0*ju + a0*dju)
        prices.append(a0)
        dprices.append(d0)
        prices.append(a0*jd)
        dprices.append(d0*jd + a0*djd)

        for j in range(1, len(A_t[i-1])):
            prices.append(A_t[i-1][j] * jd)
            dprices.append(dA_t[i-1][j] * jd + A_t[i-1][j] * djd)

        A_t.append(prices)
        dA_t.append(dprices)

    mul = np.exp(-r*delta_t)
    X = []
    dX = []
    Xet = []
    dXet = []

    for i in range(len(A_t[num_steps])):
        Xet.append(max(A_t[num_steps][i] - B, 0))
        if(A_t[num_steps][i] <= B):
            dXet.append(0)
        else:
            dXet.append(dA_t[num_steps][i])
    
    X.append(Xet)
    dX.append(dXet)

    for i in range(1, num_steps+1):
        Xt = []
        dXt = []
        for j in range(1, len(X[i-1])-1):
            xtj = X[i-1][j-1] * pu
            dxtj = (dX[i-1][j-1] * pu + X[i-1][j-1] * dpu)
            xtj += ((X[i-1][j] * 1) - (pu * X[i-1][j]) - (pd * X[i-1][j]))
            dxtj += (dX[i-1][j] - (pu * dX[i-1][j] + dpu * X[i-1][j]) - (pd * dX[i-1][j] + dpd * X[i-1][j]))
            xtj += X[i-1][j+1] * pd
            dxtj += (dX[i-1][j+1] * pd + X[i-1][j+1] * dpd)
            xtj *= mul
            dxtj *= mul
            Xt.append(xtj)
            dXt.append(dxtj)
        X.append(Xt)
        dX.append(dXt)

    return dX[num_steps][0]
